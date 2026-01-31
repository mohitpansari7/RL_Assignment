"""
PART 2: Markov Decision Process (MDP) for Retail Chatbot
==========================================================

This module implements the MDP environment for training a retail chatbot.

KEY CONCEPTS:
------------
1. State Space (S): Current conversation context
2. Action Space (A): Chatbot responses
3. Transition Dynamics (P): How customer reacts to actions
4. Reward Function (R): Feedback signal for learning
5. Discount Factor (γ): Weight for future rewards
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from dataclasses import dataclass
import json


@dataclass
class State:
    """
    State representation for retail chatbot MDP.
    
    MARKOV PROPERTY:
    ---------------
    This state contains ALL information needed to make optimal decisions.
    No need to look at conversation history before this point!
    
    Components:
    ----------
    customer_segment: 'new', 'regular', or 'vip'
        - Affects discount tolerance and expectations
        
    product_category: 'Electronics', 'Clothing', 'Home', 'Sports'
        - Determines which products to recommend
        
    conversation_turn: int (0 to 10)
        - How many turns into conversation
        - Affects urgency and patience
        
    customer_satisfaction: float [0, 1]
        - 0 = very unhappy, 1 = very happy
        - Drives conversation outcome
        
    mentioned_issue: bool
        - Did customer mention a problem/complaint?
        - Affects whether to offer discount
        
    asked_about_price: bool
        - Is customer price-sensitive?
        - Affects discount strategy
    """
    customer_segment: str  # 'new', 'regular', 'vip'
    product_category: str  # 'Electronics', 'Clothing', 'Home', 'Sports'
    conversation_turn: int  # 0 to 10
    customer_satisfaction: float  # [0, 1]
    mentioned_issue: bool  # Customer complained?
    asked_about_price: bool  # Customer price-sensitive?
    
    def to_dict(self) -> Dict:
        """Convert state to dictionary for logging."""
        return {
            'customer_segment': self.customer_segment,
            'product_category': self.product_category,
            'conversation_turn': self.conversation_turn,
            'customer_satisfaction': self.customer_satisfaction,
            'mentioned_issue': self.mentioned_issue,
            'asked_about_price': self.asked_about_price
        }
    
    def __repr__(self):
        return f"State(segment={self.customer_segment}, category={self.product_category}, " \
               f"turn={self.conversation_turn}, satisfaction={self.customer_satisfaction:.2f})"


class ActionSpace:
    """
    Defines all possible actions the chatbot can take.
    
    DESIGN CHOICE:
    -------------
    We simplify to 6 action types (instead of 75) for learning purposes.
    In production, you'd have many more specific actions.
    """
    
    # Action indices
    RECOMMEND_BUDGET = 0      # Recommend cheap product
    RECOMMEND_PREMIUM = 1     # Recommend expensive/quality product
    ANSWER_FAQ = 2            # Answer general question
    OFFER_DISCOUNT_10 = 3     # Offer 10% discount
    OFFER_DISCOUNT_20 = 4     # Offer 20% discount
    ESCALATE = 5              # Escalate to human agent
    
    ACTION_NAMES = [
        "recommend_budget",
        "recommend_premium", 
        "answer_faq",
        "offer_10%_discount",
        "offer_20%_discount",
        "escalate_to_human"
    ]
    
    @staticmethod
    def n_actions() -> int:
        """Total number of actions."""
        return len(ActionSpace.ACTION_NAMES)
    
    @staticmethod
    def get_action_name(action_id: int) -> str:
        """Get human-readable action name."""
        return ActionSpace.ACTION_NAMES[action_id]


class RetailMDP:
    """
    Retail Chatbot MDP Environment.
    
    TRANSITION DYNAMICS:
    -------------------
    Models how customers react to chatbot actions.
    P(s' | s, a) = Probability of next state given current state and action
    
    Key insight: Customer reactions are STOCHASTIC (uncertain)
    - Same action → different outcomes based on probability
    """
    
    def __init__(self, max_turns: int = 10, gamma: float = 0.99):
        """
        Initialize MDP environment.
        
        Args:
            max_turns: Maximum conversation length
            gamma: Discount factor for future rewards
        """
        self.max_turns = max_turns
        self.gamma = gamma
        self.current_state = None
        self.reset()
    
    def reset(self) -> State:
        """
        Reset environment to initial state.
        
        INITIAL STATE SAMPLING:
        ----------------------
        Randomly sample customer profile to create diverse scenarios.
        """
        self.current_state = State(
            customer_segment=np.random.choice(['new', 'regular', 'vip'], 
                                             p=[0.5, 0.35, 0.15]),
            product_category=np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports'],
                                             p=[0.4, 0.3, 0.2, 0.1]),
            conversation_turn=0,
            customer_satisfaction=np.random.uniform(0.4, 0.7),  # Start neutral-ish
            mentioned_issue=np.random.random() < 0.3,  # 30% have complaints
            asked_about_price=np.random.random() < 0.4  # 40% price-sensitive
        )
        return self.current_state
    
    def step(self, action: int) -> Tuple[State, float, bool, Dict]:
        """
        Execute one step in the environment.
        
        TRANSITION DYNAMICS:
        -------------------
        How customer reacts depends on:
        1. Action taken
        2. Current state (segment, satisfaction, etc.)
        3. Randomness (stochasticity)
        
        Args:
            action: Action index (0-5)
            
        Returns:
            next_state: New state after action
            reward: Immediate reward
            done: Episode finished?
            info: Additional information
        """
        # Copy current state
        next_state = State(
            customer_segment=self.current_state.customer_segment,
            product_category=self.current_state.product_category,
            conversation_turn=self.current_state.conversation_turn + 1,
            customer_satisfaction=self.current_state.customer_satisfaction,
            mentioned_issue=self.current_state.mentioned_issue,
            asked_about_price=self.current_state.asked_about_price
        )
        
        # Initialize tracking
        purchased = False
        escalated = False
        
        # Apply action effects on satisfaction
        satisfaction_change = self._compute_satisfaction_change(action, self.current_state)
        next_state.customer_satisfaction = np.clip(
            next_state.customer_satisfaction + satisfaction_change, 
            0.0, 1.0
        )
        
        # Check if customer purchases (stochastic based on satisfaction)
        if action in [ActionSpace.RECOMMEND_BUDGET, ActionSpace.RECOMMEND_PREMIUM]:
            purchase_prob = self._compute_purchase_probability(next_state, action)
            purchased = np.random.random() < purchase_prob
            if purchased:
                next_state.customer_satisfaction = 0.95  # Very happy after purchase
        
        # Check if escalated
        if action == ActionSpace.ESCALATE:
            escalated = True
        
        # Compute reward
        reward = self._compute_reward(
            self.current_state,
            next_state, 
            action,
            purchased,
            escalated
        )
        
        # Check if episode done
        done = (
            next_state.conversation_turn >= self.max_turns or
            purchased or
            escalated or
            next_state.customer_satisfaction < 0.2  # Customer left angry
        )
        
        # Update current state
        self.current_state = next_state
        
        info = {
            'purchased': purchased,
            'escalated': escalated,
            'satisfaction': next_state.customer_satisfaction
        }
        
        return next_state, reward, done, info
    
    def _compute_satisfaction_change(self, action: int, state: State) -> float:
        """
        Compute how satisfaction changes based on action.
        
        BEHAVIOR MODELING:
        -----------------
        Different actions affect satisfaction differently based on context.
        """
        change = 0.0
        
        if action == ActionSpace.RECOMMEND_BUDGET:
            # Budget recommendation
            if state.asked_about_price:
                change = 0.2  # Price-sensitive customer likes this
            else:
                change = 0.05  # Others neutral
                
        elif action == ActionSpace.RECOMMEND_PREMIUM:
            # Premium recommendation
            if state.customer_segment == 'vip':
                change = 0.25  # VIPs love premium
            elif state.mentioned_issue:
                change = 0.2  # Good for unhappy customers (quality)
            else:
                change = 0.1
                
        elif action == ActionSpace.ANSWER_FAQ:
            change = 0.1  # Helpful, but not exciting
            
        elif action == ActionSpace.OFFER_DISCOUNT_10:
            if state.mentioned_issue:
                change = 0.3  # Great for unhappy customers
            else:
                change = 0.15
                
        elif action == ActionSpace.OFFER_DISCOUNT_20:
            if state.mentioned_issue:
                change = 0.4  # Even better for complaints
            else:
                change = 0.25
                
        elif action == ActionSpace.ESCALATE:
            if state.customer_satisfaction < 0.4:
                change = 0.1  # Good if customer frustrated
            else:
                change = -0.1  # Annoying if unnecessary
        
        # Add noise (stochasticity)
        noise = np.random.normal(0, 0.05)
        return change + noise
    
    def _compute_purchase_probability(self, state: State, action: int) -> float:
        """
        Compute probability customer purchases.
        
        PROBABILISTIC OUTCOMES:
        ----------------------
        Higher satisfaction → higher purchase probability
        Premium products → lower purchase probability (expensive)
        """
        base_prob = state.customer_satisfaction
        
        if action == ActionSpace.RECOMMEND_BUDGET:
            return base_prob * 0.8  # Easier to sell cheap items
        elif action == ActionSpace.RECOMMEND_PREMIUM:
            if state.customer_segment == 'vip':
                return base_prob * 0.7
            else:
                return base_prob * 0.5  # Harder to sell expensive items
        
        return 0.0
    
    def _compute_reward(self, old_state: State, new_state: State, 
                       action: int, purchased: bool, escalated: bool) -> float:
        """
        Compute reward for this transition.
        
        REWARD FUNCTION:
        ---------------
        R(s,a) = w1 * satisfaction_gain + w2 * purchase + w3 * escalation_penalty
        
        Weights (business priorities):
        - w1 = 0.5  (customer satisfaction)
        - w2 = 0.35 (revenue from purchases)
        - w3 = -0.15 (cost of escalation)
        """
        # Satisfaction gain
        satisfaction_gain = new_state.customer_satisfaction - old_state.customer_satisfaction
        
        # Purchase indicator
        purchase_indicator = 1.0 if purchased else 0.0
        
        # Escalation penalty
        escalation_penalty = 1.0 if escalated else 0.0
        
        # Weighted sum
        reward = (
            0.5 * satisfaction_gain +
            0.35 * purchase_indicator +
            (-0.15) * escalation_penalty
        )
        
        return reward


if __name__ == "__main__":
    print("=" * 60)
    print("Testing MDP Environment")
    print("=" * 60)
    
    # Create environment
    env = RetailMDP()
    
    # Test one episode
    print("\nRunning sample episode with random actions:")
    state = env.reset()
    print(f"\nInitial state: {state}")
    
    total_reward = 0
    for turn in range(5):
        action = np.random.randint(0, ActionSpace.n_actions())
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        
        print(f"\nTurn {turn + 1}:")
        print(f"  Action: {ActionSpace.get_action_name(action)}")
        print(f"  Reward: {reward:.3f}")
        print(f"  New satisfaction: {info['satisfaction']:.2f}")
        print(f"  Purchased: {info['purchased']}, Escalated: {info['escalated']}")
        
        if done:
            print(f"\nEpisode finished!")
            break
    
    print(f"\nTotal reward: {total_reward:.3f}")
    
    print("\n" + "=" * 60)
    print("Running 1000 Random Episodes (Baseline)")
    print("=" * 60)
    
    # Evaluate random policy
    n_episodes = 1000
    total_rewards = []
    purchase_count = 0
    escalation_count = 0
    avg_turns = []
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        turns = 0
        
        while True:
            action = np.random.randint(0, ActionSpace.n_actions())
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            turns += 1
            
            if info['purchased']:
                purchase_count += 1
            if info['escalated']:
                escalation_count += 1
            
            if done:
                break
        
        total_rewards.append(episode_reward)
        avg_turns.append(turns)
    
    print(f"\nRandom Policy Results (1000 episodes):")
    print(f"  Average Reward: {np.mean(total_rewards):.3f} ± {np.std(total_rewards):.3f}")
    print(f"  Purchase Rate: {purchase_count/n_episodes*100:.1f}%")
    print(f"  Escalation Rate: {escalation_count/n_episodes*100:.1f}%")
    print(f"  Avg Conversation Length: {np.mean(avg_turns):.1f} turns")
    
    # Save results
    results = {
        'random_policy': {
            'avg_reward': float(np.mean(total_rewards)),
            'std_reward': float(np.std(total_rewards)),
            'purchase_rate': float(purchase_count/n_episodes),
            'escalation_rate': float(escalation_count/n_episodes),
            'avg_turns': float(np.mean(avg_turns))
        }
    }
    
    with open('mdp_baseline.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to mdp_baseline.json")


def extract_state_features(state: State) -> np.ndarray:
    """
    Convert state to feature vector for value function approximation.
    
    FEATURE ENGINEERING:
    -------------------
    φ(s) = feature vector that represents state numerically
    
    We create features that capture important patterns:
    1. Customer satisfaction (continuous)
    2. Is VIP? (binary)
    3. Is regular? (binary)  
    4. Category one-hot encoding (4 features)
    5. Mentioned issue? (binary)
    6. Asked about price? (binary)
    7. Conversation progress (normalized turn count)
    
    Total: 10 features
    """
    features = []
    
    # Satisfaction (most important!)
    features.append(state.customer_satisfaction)
    
    # Customer segment (one-hot)
    features.append(1.0 if state.customer_segment == 'vip' else 0.0)
    features.append(1.0 if state.customer_segment == 'regular' else 0.0)
    # 'new' is implied when both are 0
    
    # Product category (one-hot)
    features.append(1.0 if state.product_category == 'Electronics' else 0.0)
    features.append(1.0 if state.product_category == 'Clothing' else 0.0)
    features.append(1.0 if state.product_category == 'Home' else 0.0)
    features.append(1.0 if state.product_category == 'Sports' else 0.0)
    
    # Context flags
    features.append(1.0 if state.mentioned_issue else 0.0)
    features.append(1.0 if state.asked_about_price else 0.0)
    
    # Conversation progress (normalized 0-1)
    features.append(state.conversation_turn / 10.0)
    
    return np.array(features)


def train_value_function(env: RetailMDP, n_episodes: int = 1000):
    """
    Learn V(s) ≈ θᵀ·φ(s) using linear regression.
    
    ALGORITHM:
    ---------
    1. Collect episodes with random policy
    2. For each state, compute actual cumulative reward (Monte Carlo return)
    3. Fit linear model: V(s) = θᵀ·φ(s)
    
    This gives us value estimates for states!
    """
    print("\n" + "=" * 60)
    print("Training Value Function Approximation")
    print("=" * 60)
    
    # Collect data
    states_features = []
    returns = []
    
    for episode in range(n_episodes):
        state = env.reset()
        trajectory = []  # [(state, reward)]
        
        while True:
            action = np.random.randint(0, ActionSpace.n_actions())
            next_state, reward, done, info = env.step(action)
            trajectory.append((state, reward))
            state = next_state
            
            if done:
                break
        
        # Compute returns (cumulative discounted rewards)
        G = 0
        for i in range(len(trajectory) - 1, -1, -1):
            state, reward = trajectory[i]
            G = reward + env.gamma * G
            
            # Store (features, return) pair
            features = extract_state_features(state)
            states_features.append(features)
            returns.append(G)
    
    # Convert to numpy arrays
    X = np.array(states_features)  # (n_samples, 10 features)
    y = np.array(returns)          # (n_samples,)
    
    # Fit linear model: V(s) = θᵀ·φ(s)
    # Using closed-form solution: θ = (XᵀX)⁻¹Xᵀy
    theta = np.linalg.lstsq(X, y, rcond=None)[0]
    
    # Evaluate fit quality
    predictions = X @ theta
    mse = np.mean((predictions - y) ** 2)
    r2 = 1 - (np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2))
    
    print(f"\n✓ Value function trained on {len(returns)} state samples")
    print(f"  Mean Squared Error: {mse:.4f}")
    print(f"  R² Score: {r2:.4f}")
    print(f"  Feature weights θ: {theta}")
    
    # Test predictions
    print(f"\n  Sample Predictions:")
    test_state = State('vip', 'Electronics', 3, 0.8, False, False)
    features = extract_state_features(test_state)
    value = features @ theta
    print(f"    State: VIP, Electronics, turn=3, satisfaction=0.8")
    print(f"    Predicted V(s): {value:.3f}")
    
    test_state = State('new', 'Sports', 1, 0.3, True, True)
    features = extract_state_features(test_state)
    value = features @ theta
    print(f"    State: New, Sports, turn=1, satisfaction=0.3, issue=True")
    print(f"    Predicted V(s): {value:.3f}")
    
    return theta


if __name__ == "__main__":
    print("=" * 60)
    print("Testing MDP Environment")
    print("=" * 60)
    
    # Create environment
    env = RetailMDP()
    
    # Test one episode
    print("\nRunning sample episode with random actions:")
    state = env.reset()
    print(f"\nInitial state: {state}")
    
    total_reward = 0
    for turn in range(5):
        action = np.random.randint(0, ActionSpace.n_actions())
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        
        print(f"\nTurn {turn + 1}:")
        print(f"  Action: {ActionSpace.get_action_name(action)}")
        print(f"  Reward: {reward:.3f}")
        print(f"  New satisfaction: {info['satisfaction']:.2f}")
        print(f"  Purchased: {info['purchased']}, Escalated: {info['escalated']}")
        
        if done:
            print(f"\nEpisode finished!")
            break
    
    print(f"\nTotal reward: {total_reward:.3f}")
    
    print("\n" + "=" * 60)
    print("Running 1000 Random Episodes (Baseline)")
    print("=" * 60)
    
    # Evaluate random policy
    n_episodes = 1000
    total_rewards = []
    purchase_count = 0
    escalation_count = 0
    avg_turns = []
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        turns = 0
        
        while True:
            action = np.random.randint(0, ActionSpace.n_actions())
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            turns += 1
            
            if info['purchased']:
                purchase_count += 1
            if info['escalated']:
                escalation_count += 1
            
            if done:
                break
        
        total_rewards.append(episode_reward)
        avg_turns.append(turns)
    
    print(f"\nRandom Policy Results (1000 episodes):")
    print(f"  Average Reward: {np.mean(total_rewards):.3f} ± {np.std(total_rewards):.3f}")
    print(f"  Purchase Rate: {purchase_count/n_episodes*100:.1f}%")
    print(f"  Escalation Rate: {escalation_count/n_episodes*100:.1f}%")
    print(f"  Avg Conversation Length: {np.mean(avg_turns):.1f} turns")
    
    # Save results
    results = {
        'random_policy': {
            'avg_reward': float(np.mean(total_rewards)),
            'std_reward': float(np.std(total_rewards)),
            'purchase_rate': float(purchase_count/n_episodes),
            'escalation_rate': float(escalation_count/n_episodes),
            'avg_turns': float(np.mean(avg_turns))
        }
    }
    
    with open('mdp_baseline.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to mdp_baseline.json")
    
    # Train value function
    theta = train_value_function(env, n_episodes=1000)
    
    # Save value function
    results['value_function'] = {
        'theta': theta.tolist(),
        'feature_names': [
            'satisfaction', 'is_vip', 'is_regular',
            'Electronics', 'Clothing', 'Home', 'Sports',
            'mentioned_issue', 'asked_price', 'turn_progress'
        ]
    }
    
    with open('mdp_report.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Full report saved to mdp_report.json")