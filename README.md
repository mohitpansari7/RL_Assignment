# Part 1: Data Quality (DQ) & Distributional Data Quality (DDQ)

## 📚 Learning Objectives

After completing this part, you'll understand:
1. Why data quality matters in ML pipelines
2. How to measure completeness, duplicates, and format validity
3. What distributional shift is and why it breaks models
4. How KL-Divergence quantifies distribution mismatch

---

## 🎯 Core Concepts

### 1. Data Quality (DQ)

**Definition**: Measures if individual data points are correct.

**Why it matters**: 
- Garbage in = garbage out
- Missing values → model can't learn patterns
- Duplicates → model overfits to repeated examples
- Invalid formats → data pipeline crashes

**Metrics**:

| Metric | Formula | Threshold | Why? |
|--------|---------|-----------|------|
| Completeness | `1 - (missing/total)` | ≥ 0.95 | Need 95%+ data present |
| Duplicate Rate | `1 - (duplicates/total)` | ≥ 0.98 | Max 2% duplicates |
| Format Validity | `valid/total` | ≥ 0.99 | 99%+ correctly formatted |

**Example**:
```
Dataset: 10,000 conversations
Missing values: 500 → Completeness = 1 - (500/10000) = 0.95 ✓ PASS
Duplicates: 300 → Duplicate Rate = 1 - (300/10000) = 0.97 ✗ FAIL (need ≥ 0.98)
```

---

### 2. Distributional Data Quality (DDQ)

**Definition**: Measures if your training data matches production data.

**The Problem**:
Imagine you train a model on:
- 90% Electronics
- 5% Clothing
- 5% Other

But in production, customers actually buy:
- 40% Electronics
- 40% Clothing
- 20% Other

→ Your model will be biased toward electronics and fail on clothing!

**Solution**: KL-Divergence

---

### 3. KL-Divergence (Kullback-Leibler Divergence)

**Intuition**: "How surprised would we be if we used training distribution to model production?"

**Formula**:
```
KL(P || Q) = Σ P(x) * log(P(x) / Q(x))

Where:
- P(x) = production distribution (real-world)
- Q(x) = training distribution (our data)
```

**Properties**:
- KL = 0 → Distributions identical (perfect!)
- KL > 0 → Distributions different
- KL is NOT symmetric: KL(P||Q) ≠ KL(Q||P)

**Example Calculation**:

Let's say we have product categories:

| Category | Production P(x) | Training Q(x) | P(x) * log(P(x)/Q(x)) |
|----------|----------------|---------------|----------------------|
| Electronics | 0.40 | 0.90 | 0.40 * log(0.40/0.90) = -0.328 |
| Clothing | 0.40 | 0.05 | 0.40 * log(0.40/0.05) = +0.832 |
| Other | 0.20 | 0.05 | 0.20 * log(0.20/0.05) = +0.277 |

**KL = -0.328 + 0.832 + 0.277 = 0.781**

This is HIGH (> 0.1 threshold) → distributions are very different!

**Interpretation**:
- KL < 0.1 → Safe to use (distributions similar)
- KL > 0.1 → Dangerous (model will fail in production)

---

## 🔧 Implementation Details

### Data Generation

We create **synthetic data** with intentional quality issues:

```python
# Introduce 20% missing values
# Introduce 5% duplicates
# Introduce 3% invalid timestamps
# Imbalanced categories: 60% Electronics, 30% Clothing, 7% Home, 3% Sports
```

**Why synthetic?** 
- For learning, we need controlled problems
- Real-world data has legal/privacy restrictions

### Data Cleaning Strategy

```python
1. Remove duplicates (keep first occurrence)
2. Fix invalid timestamps (drop malformed dates)
3. Handle missing values (drop rows with NaN)
```

**Trade-off**: We lose ~25% of data, but gain quality!

---

## 🏃 Running the Code

```bash
cd /home/claude/rlhf_retail_chatbot/part1_dq_ddq
python dq_ddq.py
```

**Output**:
- `quality_report.json` - DQ and DDQ metrics
- `cleaned_data.csv` - Cleaned dataset for Parts 3-5

**Expected Results**:

Before cleaning:
- Completeness: ~0.80 (FAIL)
- Duplicate Rate: ~0.95 (FAIL)
- Format Validity: ~0.97 (FAIL)

After cleaning:
- Completeness: 1.00 ✓
- Duplicate Rate: 1.00 ✓
- Format Validity: 1.00 ✓

DDQ (comparing to production):
- Category Balance KL: ~0.45 (FAIL - very imbalanced)
- Sentiment Distribution KL: ~0.05 (PASS)
- Temporal Consistency KL: ~0.03 (PASS)

---

## 🤔 Key Insights

### Why DDQ Fails on Category Balance

Our training data:
- 60% Electronics, 30% Clothing, 7% Home, 3% Sports

Production data (ideal):
- 25% each category

**KL-Divergence**: ~0.45 (very high!)

**What this means**:
- Model will recommend electronics too often
- Poor performance on sports/home categories
- Real-world accuracy will drop

**Solutions**:
1. **Resampling**: Oversample minority categories
2. **Class weights**: Weight loss function by category
3. **Collect more data**: Get more sports/home examples

### Why This Matters for Chatbots

If your chatbot trains on biased data:
- Electronics customers → great experience
- Sports customers → poor recommendations
- Business loses money from 75% of customers!

---

## 📊 Understanding the Report

The `quality_report.json` contains:

```json
{
  "dq_assessment": {
    "metrics": {
      "completeness": {
        "value": 1.0,
        "pass": true
      },
      ...
    },
    "status": "PASS"
  },
  "ddq_assessment": {
    "metrics": {
      "category_balance": {
        "kl_divergence": 0.45,
        "pass": false
      },
      ...
    },
    "status": "FAIL"
  },
  "overall_status": "FAIL"
}
```

**Interpretation**: Data quality is good (DQ passed), but distribution is wrong (DDQ failed).

---

## 🎓 Grading Criteria (15 marks)

- **Data Generation** (3 marks): Realistic quality issues?
- **DQ Metrics** (5 marks): Correct calculations?
- **DDQ Implementation** (4 marks): KL-Divergence for ≥3 features?
- **Reporting** (3 marks): Clear JSON with pass/fail?

---

## 🔗 Connection to Next Parts

**Clean data** from Part 1 feeds into:
- **Part 3**: Pretraining the language model
- **Part 4**: Supervised fine-tuning
- **Part 5**: RLHF training

Without quality data, all subsequent training fails!

---

## 💡 Real-World Applications

This same DQ/DDQ framework applies to:
- Fraud detection (are training frauds similar to production frauds?)
- Medical diagnosis (training data vs hospital population)
- Recommendation systems (training users vs actual users)

**Industry Standard**: Netflix, Amazon, Google all use KL-Divergence to monitor data drift!

## Limitations

Part 1 identified significant DDQ failures:
- Category imbalance (KL=0.60) will bias model toward electronics
- Temporal drift (KL=0.23) may reduce accuracy on recent trends

In production, we would:
1. Oversample minority categories (Home/Sports)
2. Collect more 2024 data
3. Use stratified sampling to match production distribution

For this learning assignment, we proceed with cleaned data
while acknowledging these limitations will affect model performance.


# Part 2: Markov Decision Process (MDP)

## What We Built

A complete MDP simulator for retail chatbot training with:
- **State space**: Customer context (segment, category, satisfaction, etc.)
- **Action space**: 6 actions (recommend, discount, FAQ, escalate)
- **Transition dynamics**: Probabilistic customer reactions
- **Reward function**: Multi-objective (satisfaction + purchase - escalation)
- **Value function**: V(s) approximation using linear regression

## Key Concepts

### MDP = Framework for Sequential Decision-Making

**5 Components:**
1. States (S): Where we are
2. Actions (A): What we can do
3. Transitions (P): What happens next
4. Rewards (R): How good was that action
5. Discount (γ): How much we care about future

### Why Simulator?
- Safe exploration (no real customer harm)
- Infinite data generation
- Models uncertainty (same action → different outcomes)

### Value Function
V(s) = Expected cumulative reward from state s

Uses 10 features:
- Satisfaction (most important)
- Customer segment (VIP/regular/new)
- Product category
- Context flags (issue, price-sensitive)

## Results

**Random Policy Baseline:**
- Avg Reward: ~0.15
- Purchase Rate: ~15-20%
- Escalation Rate: ~15-20%

**This is what RL will improve in Part 5!**

## Files
- `mdp.py`: Complete MDP implementation
- `mdp_report.json`: Baseline metrics + value function weights