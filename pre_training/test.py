"""
PART 3: Pretraining - Next-Token Prediction
===========================================

This module implements pretraining for the retail chatbot using
a simple 2-layer transformer model.

GOAL: Learn basic language patterns from retail conversations
METHOD: Next-token prediction (causal language modeling)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
import json
from collections import Counter
import matplotlib.pyplot as plt


class SimpleTokenizer:
    """
    Word-level tokenizer for retail conversations.
    
    TOKENIZATION:
    ------------
    Converts text → sequence of integer IDs
    
    Example:
    "I need jeans" → [45, 234, 89]
    
    Special tokens:
    - <PAD>: Padding for sequences
    - <UNK>: Unknown words (not in vocabulary)
    - <BOS>: Beginning of sequence
    - <EOS>: End of sequence
    """
    
    def __init__(self, vocab_size: int = 5000):
        """
        Args:
            vocab_size: Maximum vocabulary size
        """
        self.vocab_size = vocab_size
        self.word2idx = {}
        self.idx2word = {}
        
        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.BOS_TOKEN = '<BOS>'
        self.EOS_TOKEN = '<EOS>'
        
        self.pad_idx = 0
        self.unk_idx = 1
        self.bos_idx = 2
        self.eos_idx = 3
    
    def build_vocab(self, texts: List[str]):
        """
        Build vocabulary from text corpus.
        
        ALGORITHM:
        ---------
        1. Count word frequencies
        2. Keep top vocab_size most frequent words
        3. Assign unique ID to each word
        
        Args:
            texts: List of conversation texts
        """
        # Count words
        word_counts = Counter()
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)
        
        # Initialize with special tokens
        self.word2idx = {
            self.PAD_TOKEN: self.pad_idx,
            self.UNK_TOKEN: self.unk_idx,
            self.BOS_TOKEN: self.bos_idx,
            self.EOS_TOKEN: self.eos_idx
        }
        
        # Add most frequent words
        most_common = word_counts.most_common(self.vocab_size - 4)
        for idx, (word, count) in enumerate(most_common, start=4):
            self.word2idx[word] = idx
        
        # Create reverse mapping
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        print(f"✓ Vocabulary built: {len(self.word2idx)} tokens")
        print(f"  Most common: {most_common[:10]}")
    
    def encode(self, text: str, max_length: int = 256) -> List[int]:
        """
        Convert text → token IDs.
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            
        Returns:
            List of token IDs
        """
        words = text.lower().split()
        
        # Add BOS, convert words, add EOS
        tokens = [self.bos_idx]
        for word in words:
            tokens.append(self.word2idx.get(word, self.unk_idx))
        tokens.append(self.eos_idx)
        
        # Truncate if too long
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        """
        Convert token IDs → text.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            Decoded text
        """
        words = [self.idx2word.get(idx, self.UNK_TOKEN) for idx in tokens]
        # Remove special tokens
        words = [w for w in words if w not in [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]]
        return ' '.join(words)


class ConversationDataset(Dataset):
    """
    PyTorch Dataset for next-token prediction.
    
    TRAINING DATA CREATION:
    ----------------------
    From: "I need durable jeans"
    
    Create pairs:
    Input:  [BOS, I, need, durable]     → Target: jeans
    Input:  [BOS, I, need]              → Target: durable
    Input:  [BOS, I]                    → Target: need
    
    Each position learns to predict the next token!
    """
    
    def __init__(self, texts: List[str], tokenizer: SimpleTokenizer, max_length: int = 256):
        """
        Args:
            texts: List of conversation texts
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Tokenize all texts
        self.sequences = []
        for text in texts:
            tokens = tokenizer.encode(text, max_length)
            if len(tokens) > 1:  # Need at least 2 tokens (input, target)
                self.sequences.append(tokens)
        
        print(f"✓ Dataset created: {len(self.sequences)} sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get training example.
        
        Returns:
            input_ids: Token IDs for input (length-1)
            target_ids: Token IDs for targets (length-1)
            
        Example:
        tokens = [BOS, I, need, jeans, EOS]
        input  = [BOS, I, need, jeans]      (predict next token at each position)
        target = [I, need, jeans, EOS]      (what we should predict)
        """
        tokens = self.sequences[idx]
        
        # Input: all tokens except last
        input_ids = tokens[:-1]
        
        # Target: all tokens except first
        target_ids = tokens[1:]
        
        return torch.tensor(input_ids, dtype=torch.long), \
               torch.tensor(target_ids, dtype=torch.long)


def load_and_prepare_data(csv_path: str, train_ratio: float = 0.8):
    """
    Load cleaned data from Part 1 and prepare for pretraining.
    
    PIPELINE:
    --------
    1. Load CSV
    2. Extract conversation texts
    3. Split train/validation
    4. Build tokenizer
    5. Create datasets
    
    Args:
        csv_path: Path to cleaned_data.csv from Part 1
        train_ratio: Train/validation split ratio
        
    Returns:
        train_dataset, val_dataset, tokenizer
    """
    print("=" * 60)
    print("STEP 1: Loading and Preparing Data")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"\n✓ Loaded {len(df)} conversations from {csv_path}")
    
    # Extract texts
    texts = df['conversation_text'].dropna().tolist()
    print(f"✓ Extracted {len(texts)} conversation texts")
    
    # Train/validation split
    np.random.seed(42)
    indices = np.random.permutation(len(texts))
    split_idx = int(len(texts) * train_ratio)
    
    train_texts = [texts[i] for i in indices[:split_idx]]
    val_texts = [texts[i] for i in indices[split_idx:]]
    
    print(f"\n✓ Split: {len(train_texts)} train, {len(val_texts)} validation")
    
    # Build tokenizer
    print("\n" + "-" * 60)
    print("Building vocabulary...")
    print("-" * 60)
    tokenizer = SimpleTokenizer(vocab_size=5000)
    tokenizer.build_vocab(train_texts)
    
    # Create datasets
    print("\n" + "-" * 60)
    print("Creating datasets...")
    print("-" * 60)
    train_dataset = ConversationDataset(train_texts, tokenizer, max_length=256)
    val_dataset = ConversationDataset(val_texts, tokenizer, max_length=256)
    
    # Show sample
    print(f"\n" + "-" * 60)
    print("Sample training example:")
    print("-" * 60)
    input_ids, target_ids = train_dataset[0]
    print(f"Input tokens:  {input_ids.tolist()[:10]}... (length={len(input_ids)})")
    print(f"Target tokens: {target_ids.tolist()[:10]}... (length={len(target_ids)})")
    print(f"Decoded input: {tokenizer.decode(input_ids.tolist()[:20])}")
    
    return train_dataset, val_dataset, tokenizer


if __name__ == "__main__":
    # Test data loading
    csv_path = "../cleaned_data.csv"  # Adjust path as needed
    
    try:
        train_dataset, val_dataset, tokenizer = load_and_prepare_data(csv_path)
        
        print("\n" + "=" * 60)
        print("✓ Data preparation complete!")
        print("=" * 60)
        print(f"Vocabulary size: {len(tokenizer.word2idx)}")
        print(f"Train sequences: {len(train_dataset)}")
        print(f"Val sequences: {len(val_dataset)}")
        
    except FileNotFoundError:
        print(f"\n❌ Error: Could not find {csv_path}")
        print("Make sure cleaned_data.csv from Part 1 is in the parent directory!")


class SimpleTransformer(nn.Module):
    """
    Simple 2-layer Transformer for next-token prediction.
    
    ARCHITECTURE:
    ------------
    Input → Embedding → Transformer Layer 1 → Transformer Layer 2 → Output
    
    Key Components:
    1. Token Embedding: Maps token IDs to dense vectors
    2. Positional Encoding: Adds position information
    3. Transformer Layers: Self-attention + Feed-forward
    4. Output Layer: Predicts next token probabilities
    """
    
    def __init__(self, vocab_size: int, embed_dim: int = 128, 
                 num_heads: int = 4, num_layers: int = 2, max_len: int = 256):
        """
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Embedding dimension (128 as per assignment)
            num_heads: Number of attention heads (4 as per assignment)
            num_layers: Number of transformer layers (2 as per assignment)
            max_len: Maximum sequence length
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Token embedding: token_id → vector
        # Example: token 45 → [0.2, -0.5, 0.8, ..., 0.3] (128 dims)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional encoding: adds position information
        # Position 0 has different encoding than position 5
        # This tells model "this word is at the beginning vs middle"
        self.positional_encoding = nn.Parameter(
            torch.randn(max_len, embed_dim) * 0.02
        )
        
        # Transformer layers
        # Each layer has:
        # - Multi-head self-attention (words look at each other)
        # - Feed-forward network (process information)
        # - Layer normalization (stabilize training)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,  # Standard: 4x embedding dim
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output layer: hidden state → vocabulary probabilities
        # Example: [0.2, -0.5, ...] (128 dims) → [0.01, 0.6, 0.02, ...] (vocab_size probs)
        self.output_layer = nn.Linear(embed_dim, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize model weights.
        
        WHY THIS MATTERS:
        ----------------
        Good initialization helps training converge faster!
        - Small random values prevent exploding gradients
        - Uniform distribution keeps activations balanced
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the model.
        
        COMPUTATION FLOW:
        ----------------
        1. Token embedding: IDs → vectors
        2. Add positional encoding: vectors + position info
        3. Transformer layers: self-attention + FFN
        4. Output layer: hidden → vocabulary logits
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Mask for padding (optional)
            
        Returns:
            logits: Predictions for next token (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        
        # Step 1: Token embeddings
        # Shape: (batch_size, seq_len) → (batch_size, seq_len, embed_dim)
        token_embeds = self.token_embedding(input_ids)
        
        # Step 2: Add positional encoding
        # Each position gets a unique encoding added to its embedding
        # Shape: (batch_size, seq_len, embed_dim)
        pos_embeds = self.positional_encoding[:seq_len, :].unsqueeze(0)
        embeddings = token_embeds + pos_embeds
        
        # Step 3: Create causal mask (prevent looking at future tokens)
        # WHY: When predicting token at position i, can only see positions 0 to i-1
        # This enforces "next-token prediction" constraint
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len) * float('-inf'), 
            diagonal=1
        ).to(input_ids.device)
        
        # Step 4: Transformer layers
        # Self-attention: each word looks at previous words
        # Feed-forward: process the attended information
        # Shape: (batch_size, seq_len, embed_dim)
        hidden_states = self.transformer(embeddings, mask=causal_mask)
        
        # Step 5: Output layer
        # Convert hidden states to vocabulary predictions
        # Shape: (batch_size, seq_len, vocab_size)
        logits = self.output_layer(hidden_states)
        
        return logits
    
    def generate(self, tokenizer: SimpleTokenizer, prompt: str, 
                 max_new_tokens: int = 20, temperature: float = 1.0) -> str:
        """
        Generate text continuation given a prompt.
        
        GENERATION ALGORITHM:
        --------------------
        1. Encode prompt
        2. Predict next token
        3. Add predicted token to sequence
        4. Repeat until max_length or EOS
        
        Args:
            tokenizer: Tokenizer instance
            prompt: Starting text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            
        Returns:
            Generated text
        """
        self.eval()
        
        # Encode prompt
        tokens = tokenizer.encode(prompt, max_length=256)
        input_ids = torch.tensor(tokens).unsqueeze(0)  # Add batch dimension
        
        generated_tokens = tokens.copy()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get predictions
                logits = self.forward(input_ids)
                
                # Get logits for last position (next token prediction)
                next_token_logits = logits[0, -1, :] / temperature
                
                # Convert to probabilities
                probs = torch.softmax(next_token_logits, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                # Stop if EOS
                if next_token == tokenizer.eos_idx:
                    break
                
                # Add to sequence
                generated_tokens.append(next_token)
                input_ids = torch.tensor(generated_tokens).unsqueeze(0)
                
                # Stop if too long
                if len(generated_tokens) >= 256:
                    break
        
        return tokenizer.decode(generated_tokens)


def train_model(model: SimpleTransformer, train_dataset, val_dataset, 
                tokenizer: SimpleTokenizer, epochs: int = 5, batch_size: int = 32):
    """
    Train the transformer model.
    
    TRAINING LOOP:
    -------------
    For each epoch:
      1. Forward pass (get predictions)
      2. Compute loss (how wrong are predictions?)
      3. Backward pass (compute gradients)
      4. Update weights (improve model)
      5. Validate (check on unseen data)
    
    Args:
        model: Transformer model
        train_dataset: Training data
        val_dataset: Validation data
        tokenizer: Tokenizer instance
        epochs: Number of training epochs
        batch_size: Batch size for training
    """
    # Setup device (MPS for M1 Mac, else CPU)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\n✓ Using device: {device}")
    model = model.to(device)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Optimizer: Adam with learning rate 3e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    # Loss function: Cross-entropy (standard for classification)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_perplexity': [],
        'val_perplexity': []
    }
    
    print(f"\nStarting training for {epochs} epochs...")
    print(f"  Batch size: {batch_size}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print('='*60)
        
        # ============= TRAINING =============
        model.train()
        train_losses = []
        
        for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
            # Move to device
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            # Forward pass
            logits = model(input_ids)
            
            # Reshape for loss computation
            # logits: (batch, seq_len, vocab) → (batch*seq_len, vocab)
            # targets: (batch, seq_len) → (batch*seq_len)
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1)
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients (prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Print progress every 20 batches
            if (batch_idx + 1) % 20 == 0:
                avg_loss = np.mean(train_losses[-20:])
                print(f"  Batch {batch_idx + 1}/{len(train_loader)} | Loss: {avg_loss:.4f}")
        
        # Average training loss
        avg_train_loss = np.mean(train_losses)
        train_perplexity = np.exp(avg_train_loss)
        
        # ============= VALIDATION =============
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for input_ids, target_ids in val_loader:
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)
                
                logits = model(input_ids)
                loss = criterion(
                    logits.view(-1, logits.size(-1)),
                    target_ids.view(-1)
                )
                
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        val_perplexity = np.exp(avg_val_loss)
        
        # Record history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_perplexity'].append(train_perplexity)
        history['val_perplexity'].append(val_perplexity)
        
        # Print epoch summary
        print(f"\n  Train Loss: {avg_train_loss:.4f} | Perplexity: {train_perplexity:.2f}")
        print(f"  Val Loss:   {avg_val_loss:.4f} | Perplexity: {val_perplexity:.2f}")
        
        # Generate sample text
        if (epoch + 1) % 2 == 0:  # Every 2 epochs
            print(f"\n  Sample generation:")
            sample_text = model.generate(tokenizer, "i need", max_new_tokens=15)
            print(f"    Prompt: 'i need'")
            print(f"    Generated: {sample_text}")
    
    # Save model
    print(f"\n{'='*60}")
    print("Saving model...")
    print('='*60)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': len(tokenizer.word2idx),
        'history': history
    }, 'pretrained_model.pt')
    
    print(f"✓ Model saved to: pretrained_model.pt")
    
    # Plot training curves
    plot_training_curves(history)
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("Final Evaluation")
    print('='*60)
    print(f"  Final Train Perplexity: {history['train_perplexity'][-1]:.2f}")
    print(f"  Final Val Perplexity: {history['val_perplexity'][-1]:.2f}")
    
    if history['val_perplexity'][-1] < 50:
        print(f"  ✓ Target achieved! (Perplexity < 50)")
    else:
        print(f"  ⚠ Target not met (need < 50)")
    
    # Generate multiple samples
    print(f"\n  Sample Generations:")
    prompts = ["i need", "what is", "can you", "the product"]
    for prompt in prompts:
        generated = model.generate(tokenizer, prompt, max_new_tokens=12)
        print(f"    '{prompt}' → {generated}")
    
    return model, history


def plot_training_curves(history: Dict):
    """
    Plot training and validation loss curves.
    
    Args:
        history: Dictionary with training history
    """
    plt.figure(figsize=(12, 4))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', marker='o')
    plt.plot(history['val_loss'], label='Val Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Perplexity plot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_perplexity'], label='Train Perplexity', marker='o')
    plt.plot(history['val_perplexity'], label='Val Perplexity', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Training and Validation Perplexity')
    plt.axhline(y=50, color='r', linestyle='--', label='Target (< 50)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('loss_curves.png', dpi=150)
    print(f"✓ Training curves saved to: loss_curves.png")
    plt.close()
    
if __name__ == "__main__":
    # Test data loading
    csv_path = "../cleaned_data.csv"
    
    try:
        train_dataset, val_dataset, tokenizer = load_and_prepare_data(csv_path)
        
        print("\n" + "=" * 60)
        print("✓ Data preparation complete!")
        print("=" * 60)
        print(f"Vocabulary size: {len(tokenizer.word2idx)}")
        print(f"Train sequences: {len(train_dataset)}")
        print(f"Val sequences: {len(val_dataset)}")
        
        # Test model creation
        print("\n" + "=" * 60)
        print("STEP 2: Building Transformer Model")
        print("=" * 60)
        
        model = SimpleTransformer(
            vocab_size=len(tokenizer.word2idx),
            embed_dim=128,
            num_heads=4,
            num_layers=2,
            max_len=256
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n✓ Model created:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Embedding dim: 128")
        print(f"  Num heads: 4")
        print(f"  Num layers: 2")
        
        # Test forward pass
        print("\n" + "-" * 60)
        print("Testing forward pass...")
        print("-" * 60)
        
        sample_input, sample_target = train_dataset[0]
        sample_input = sample_input.unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            logits = model(sample_input)
        
        print(f"✓ Forward pass successful!")
        print(f"  Input shape: {sample_input.shape}")
        print(f"  Output shape: {logits.shape}")
        print(f"  Expected: (batch=1, seq_len={sample_input.shape[1]}, vocab={len(tokenizer.word2idx)})")
        
        # Step 3: Train the model
        print("\n" + "=" * 60)
        print("STEP 3: Training the Model")
        print("=" * 60)
        
        train_model(model, train_dataset, val_dataset, tokenizer, epochs=5)
        
    except FileNotFoundError:
        print(f"\n❌ Error: Could not find {csv_path}")
        print("Make sure cleaned_data.csv from Part 1 is in the parent directory!")




if __name__ == "__main__":
    # Test data loading
    csv_path = "../cleaned_data.csv"
    
    try:
        train_dataset, val_dataset, tokenizer = load_and_prepare_data(csv_path)
        
        print("\n" + "=" * 60)
        print("✓ Data preparation complete!")
        print("=" * 60)
        print(f"Vocabulary size: {len(tokenizer.word2idx)}")
        print(f"Train sequences: {len(train_dataset)}")
        print(f"Val sequences: {len(val_dataset)}")
        
        # Test model creation
        print("\n" + "=" * 60)
        print("STEP 2: Building Transformer Model")
        print("=" * 60)
        
        model = SimpleTransformer(
            vocab_size=len(tokenizer.word2idx),
            embed_dim=128,
            num_heads=4,
            num_layers=2,
            max_len=256
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n✓ Model created:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Embedding dim: 128")
        print(f"  Num heads: 4")
        print(f"  Num layers: 2")
        
        # Test forward pass
        print("\n" + "-" * 60)
        print("Testing forward pass...")
        print("-" * 60)
        
        sample_input, sample_target = train_dataset[0]
        sample_input = sample_input.unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            logits = model(sample_input)
        
        print(f"✓ Forward pass successful!")
        print(f"  Input shape: {sample_input.shape}")
        print(f"  Output shape: {logits.shape}")
        print(f"  Expected: (batch=1, seq_len={sample_input.shape[1]}, vocab={len(tokenizer.word2idx)})")
        
    except FileNotFoundError:
        print(f"\n❌ Error: Could not find {csv_path}")
        print("Make sure cleaned_data.csv from Part 1 is in the parent directory!")