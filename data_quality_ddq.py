"""
PART 1: Data Quality (DQ) & Distributional Data Quality (DDQ)
================================================================

CONCEPT EXPLANATION:
-------------------
In machine learning, "garbage in = garbage out". Before training any model,
we need to ensure our data is:
1. Complete (no missing values)
2. Unique (no duplicates)
3. Valid (correct formats)
4. Representative (matches production distribution)

This module implements DQ and DDQ metrics for retail chatbot data.

MATHEMATICAL FOUNDATIONS:
------------------------
1. Completeness: C = 1 - (N_missing / N_total)
   - Measures what % of data is present
   - Threshold: ≥ 0.95 (95%)

2. Duplicate Rate: D = 1 - (N_duplicates / N_total)
   - Measures uniqueness of records
   - Threshold: ≥ 0.98 (max 2% duplicates)

3. Format Validity: V = N_valid / N_total
   - Measures correctly formatted fields (emails, dates)
   - Threshold: ≥ 0.99 (99%)

4. KL-Divergence: KL(P||Q) = Σ P(x) * log(P(x)/Q(x))
   - Measures distribution mismatch
   - P = production distribution (real-world)
   - Q = training distribution (our data)
   - Threshold: < 0.1 for each feature
   
   INTUITION: KL-Divergence answers "How surprised would we be if we
   used training distribution Q to model production distribution P?"
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import re
from scipy.stats import entropy
from collections import Counter


class RetailDataGenerator:
    """
    Generates synthetic retail chatbot conversation data with realistic quality issues.
    
    WHY SYNTHETIC DATA?
    ------------------
    For learning purposes, we generate data that mimics real-world problems:
    - Missing values (customers abandon forms)
    - Duplicates (system logs same conversation twice)
    - Invalid formats (typos in emails)
    - Imbalanced categories (more people buy electronics than sports gear)
    """
    
    def __init__(self, n_records: int = 10000, seed: int = 42):
        """
        Args:
            n_records: Number of conversation records to generate
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        self.n_records = n_records
        
        # Product categories with realistic imbalance
        # REAL-WORLD INSIGHT: E-commerce sites sell more electronics than niche items
        self.categories = ['Electronics', 'Clothing', 'Home', 'Sports']
        self.category_weights = [0.60, 0.30, 0.07, 0.03]  # 60/30/7/3 split
        
        # Customer segments
        self.segments = ['new', 'regular', 'vip']
        self.segment_weights = [0.5, 0.35, 0.15]
        
        # Sentiment distribution
        self.sentiments = ['positive', 'neutral', 'negative']
        self.sentiment_weights = [0.4, 0.4, 0.2]
        
        # Outcomes
        self.outcomes = ['purchase', 'abandon', 'escalate']
        self.outcome_weights = [0.3, 0.6, 0.1]
        
    def generate_data(self) -> pd.DataFrame:
        """
        Generate synthetic retail conversation data with intentional quality issues.
        
        QUALITY ISSUES INTRODUCED:
        -------------------------
        1. Missing values (20%): Simulates incomplete customer profiles
        2. Duplicates (5%): Simulates system logging errors
        3. Invalid formats (3%): Simulates user input errors
        4. Imbalanced categories: Reflects real e-commerce patterns
        
        Returns:
            DataFrame with columns: customer_id, product_category, sentiment,
            timestamp, conversation_text, outcome
        """
        
        data = []
        
        for i in range(self.n_records):
            # Generate base record
            record = {
                'customer_id': f'CUST_{i:06d}',
                'product_category': np.random.choice(self.categories, p=self.category_weights),
                'sentiment': np.random.choice(self.sentiments, p=self.sentiment_weights),
                'timestamp': self._generate_timestamp(),
                'conversation_text': self._generate_conversation_text(),
                'outcome': np.random.choice(self.outcomes, p=self.outcome_weights)
            }
            
            data.append(record)
        
        df = pd.DataFrame(data)
        
        # Introduce quality issues
        df = self._introduce_missing_values(df, missing_rate=0.20)
        df = self._introduce_duplicates(df, duplicate_rate=0.05)
        df = self._introduce_invalid_formats(df, invalid_rate=0.03)
        
        return df
    
    def _generate_timestamp(self) -> str:
        """
        Generate timestamps spread over 5 years (2019-2024).
        
        WHY 5 YEARS?
        -----------
        Assignment mentions "5 years of chat logs" to simulate temporal drift
        (customer behavior changes over time, especially post-COVID 2020)
        """
        start_date = datetime(2019, 1, 1)
        end_date = datetime(2024, 12, 31)
        time_between = end_date - start_date
        random_days = np.random.randint(0, time_between.days)
        random_timestamp = start_date + timedelta(days=random_days)
        return random_timestamp.isoformat()
    
    def _generate_conversation_text(self) -> str:
        """Generate realistic customer conversation snippets."""
        templates = [
            "I'm looking for a durable product for outdoor use",
            "My previous purchase broke after 2 weeks, need replacement",
            "What's your return policy?",
            "Can you recommend something within $50-100 budget?",
            "I need this shipped urgently for a gift",
            "The product description doesn't match what I received",
            "Looking for high-rated items with good reviews"
        ]
        return np.random.choice(templates)
    
    def _introduce_missing_values(self, df: pd.DataFrame, missing_rate: float) -> pd.DataFrame:
        """
        Introduce missing values randomly across all columns.
        
        MATHEMATICAL APPROACH:
        ---------------------
        For each cell, flip a biased coin with P(missing) = missing_rate
        This creates ~20% missing values across dataset
        
        REAL-WORLD CAUSE:
        ----------------
        - Customers skip optional fields
        - System timeouts during data collection
        - Integration failures between systems
        """
        df_copy = df.copy()
        n_cells = df.shape[0] * df.shape[1]
        n_missing = int(n_cells * missing_rate)
        
        # Randomly select cells to make NaN
        for _ in range(n_missing):
            row = np.random.randint(0, df.shape[0])
            col = np.random.choice(df.columns)
            df_copy.loc[row, col] = np.nan
        
        return df_copy
    
    def _introduce_duplicates(self, df: pd.DataFrame, duplicate_rate: float) -> pd.DataFrame:
        """
        Introduce duplicate records.
        
        WHY DUPLICATES OCCUR:
        --------------------
        - Retry logic in APIs (customer clicks "submit" twice)
        - Batch processing errors
        - Database replication issues
        """
        df_copy = df.copy()
        n_duplicates = int(len(df) * duplicate_rate)
        
        # Randomly select rows to duplicate
        duplicate_indices = np.random.choice(df.index, size=n_duplicates, replace=True)
        duplicates = df_copy.loc[duplicate_indices].copy()
        
        # Append duplicates
        df_with_dupes = pd.concat([df_copy, duplicates], ignore_index=True)
        
        return df_with_dupes
    
    def _introduce_invalid_formats(self, df: pd.DataFrame, invalid_rate: float) -> pd.DataFrame:
        """
        Introduce invalid timestamp formats.
        
        CORRUPTION STRATEGY:
        -------------------
        Replace valid ISO timestamps with malformed strings
        Simulates data corruption during ETL pipelines
        """
        df_copy = df.copy()
        n_invalid = int(len(df) * invalid_rate)
        
        invalid_indices = np.random.choice(df.index, size=n_invalid, replace=False)
        
        for idx in invalid_indices:
            if 'timestamp' in df_copy.columns:
                # Corrupt the timestamp
                df_copy.loc[idx, 'timestamp'] = 'INVALID_DATE_' + str(idx)
        
        return df_copy


class DataQualityAssessor:
    """
    Assesses Data Quality (DQ) metrics: completeness, duplicates, format validity.
    
    GRADING CRITERIA:
    ----------------
    - Completeness ≥ 0.95 (95%)
    - Duplicate Rate ≥ 0.98 (max 2% duplicates)
    - Format Validity ≥ 0.99 (99%)
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: DataFrame to assess
        """
        self.df = df
        self.metrics = {}
    
    def assess_completeness(self) -> float:
        """
        Calculate completeness: 1 - (missing_values / total_values)
        
        MATHEMATICAL DERIVATION:
        -----------------------
        Total cells = n_rows × n_columns
        Missing cells = count of NaN values
        Completeness = 1 - (Missing / Total)
        
        INTERPRETATION:
        --------------
        Completeness = 0.95 → 95% of data is present, 5% missing
        """
        total_values = self.df.shape[0] * self.df.shape[1]
        missing_values = self.df.isna().sum().sum()
        completeness = 1 - (missing_values / total_values)
        
        self.metrics['completeness'] = {
            'value': float(completeness),
            'missing_values': int(missing_values),
            'total_values': int(total_values),
            'threshold': 0.95,
            'pass': bool(completeness >= 0.95)
        }
        
        return completeness
    
    def assess_duplicates(self) -> float:
        """
        Calculate duplicate rate: 1 - (duplicates / total_records)
        
        APPROACH:
        --------
        1. Identify duplicate rows (all columns match)
        2. Count duplicates
        3. Compute rate
        
        WHY THIS MATTERS:
        ----------------
        Duplicates cause:
        - Overfitting (model sees same example multiple times)
        - Biased metrics (same customer counted twice)
        - Wasted computation
        """
        total_records = len(self.df)
        duplicate_records = self.df.duplicated().sum()
        duplicate_rate = 1 - (duplicate_records / total_records)
        
        self.metrics['duplicate_rate'] = {
            'value': float(duplicate_rate),
            'duplicate_count': int(duplicate_records),
            'total_records': int(total_records),
            'threshold': 0.98,
            'pass': bool(duplicate_rate >= 0.98)
        }
        
        return duplicate_rate
    
    def assess_format_validity(self) -> float:
        """
        Calculate format validity for timestamps.
        
        VALIDATION LOGIC:
        ----------------
        Valid timestamp: ISO format (YYYY-MM-DDTHH:MM:SS)
        Invalid: anything else
        
        REGEX PATTERN:
        -------------
        ^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2} matches ISO format
        """
        if 'timestamp' not in self.df.columns:
            self.metrics['format_validity'] = {
                'value': 1.0,
                'valid_count': 0,
                'total_count': 0,
                'threshold': 0.99,
                'pass': True
            }
            return 1.0
        
        # Drop NaN timestamps for this check
        timestamps = self.df['timestamp'].dropna()
        total_count = len(timestamps)
        
        # ISO format pattern: YYYY-MM-DDTHH:MM:SS
        iso_pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}'
        
        valid_count = timestamps.astype(str).str.match(iso_pattern).sum()
        format_validity = valid_count / total_count if total_count > 0 else 0
        
        self.metrics['format_validity'] = {
            'value': float(format_validity),
            'valid_count': int(valid_count),
            'total_count': int(total_count),
            'threshold': 0.99,
            'pass': bool(format_validity >= 0.99)
        }
        
        return format_validity
    
    def get_all_metrics(self) -> Dict:
        """
        Run all DQ assessments and return consolidated report.
        
        Returns:
            Dictionary with all DQ metrics
        """
        self.assess_completeness()
        self.assess_duplicates()
        self.assess_format_validity()
        
        return self.metrics


class DistributionalQualityAssessor:
    """
    Assesses Distributional Data Quality (DDQ) using KL-Divergence.
    
    CORE CONCEPT: KL-Divergence
    ---------------------------
    KL(P||Q) = Σ P(x) * log(P(x) / Q(x))
    
    WHERE:
    - P(x) = production distribution (how feature x appears in real world)
    - Q(x) = training distribution (how feature x appears in our data)
    - KL = 0: distributions identical (perfect!)
    - KL > 0: distributions different (problematic if large)
    
    INTUITION:
    ---------
    Imagine you're training on 90% electronics data, but production has
    40% electronics. The model will be "surprised" by production data.
    KL-Divergence quantifies this surprise.
    
    ACCEPTANCE CRITERIA:
    -------------------
    KL < 0.1 for all key features (category, sentiment, segment)
    """
    
    def __init__(self, train_df: pd.DataFrame, prod_df: pd.DataFrame):
        """
        Args:
            train_df: Training dataset
            prod_df: Production dataset (simulated as "ideal" distribution)
        """
        self.train_df = train_df
        self.prod_df = prod_df
        self.metrics = {}
    
    def compute_kl_divergence(self, feature: str) -> float:
        """
        Compute KL-Divergence for a categorical feature.
        
        ALGORITHM:
        ---------
        1. Get value counts for feature in both datasets
        2. Convert to probability distributions
        3. Align categories (handle missing categories)
        4. Compute KL(P_prod || P_train)
        5. Add epsilon (1e-10) to avoid log(0)
        
        MATHEMATICAL STEPS:
        ------------------
        P_prod = production_counts / total_prod
        P_train = training_counts / total_train
        KL = Σ P_prod(x) * log(P_prod(x) / P_train(x))
        
        Args:
            feature: Column name to compute KL for
            
        Returns:
            KL-Divergence value
        """
        # Get distributions (drop NaN)
        prod_counts = self.prod_df[feature].value_counts(normalize=True, dropna=True)
        train_counts = self.train_df[feature].value_counts(normalize=True, dropna=True)
        
        # Align categories (ensure both have same keys)
        all_categories = set(prod_counts.index) | set(train_counts.index)
        
        # Add epsilon to avoid log(0)
        epsilon = 1e-10
        
        prod_probs = np.array([prod_counts.get(cat, epsilon) for cat in all_categories])
        train_probs = np.array([train_counts.get(cat, epsilon) for cat in all_categories])
        
        # Normalize to ensure they sum to 1
        prod_probs = prod_probs / prod_probs.sum()
        train_probs = train_probs / train_probs.sum()
        
        # Compute KL-Divergence: KL(P || Q) = Σ P(x) * log(P(x) / Q(x))
        kl_div = np.sum(prod_probs * np.log(prod_probs / train_probs))
        
        return float(kl_div)
    
    def assess_category_balance(self) -> float:
        """
        Assess if product categories are balanced between train and prod.
        
        WHY THIS MATTERS:
        ----------------
        If training has 90% electronics but production has 40%, model will:
        - Be biased toward electronics recommendations
        - Perform poorly on clothing/home/sports
        """
        kl = self.compute_kl_divergence('product_category')
        
        self.metrics['category_balance'] = {
            'kl_divergence': kl,
            'threshold': 0.1,
            'pass': bool(kl < 0.1)
        }
        
        return kl
    
    def assess_sentiment_distribution(self) -> float:
        """
        Assess if sentiment distribution is balanced.
        
        EXAMPLE PROBLEM:
        ---------------
        Training: 80% positive, 10% neutral, 10% negative
        Production: 40% positive, 40% neutral, 20% negative
        → Model will be overconfident about positive outcomes
        """
        kl = self.compute_kl_divergence('sentiment')
        
        self.metrics['sentiment_distribution'] = {
            'kl_divergence': kl,
            'threshold': 0.1,
            'pass': bool(kl < 0.1)
        }
        
        return kl
    
    def assess_temporal_consistency(self) -> float:
        """
        Assess if data is evenly distributed over time periods.
        
        APPROACH:
        --------
        Bin timestamps into quarters (Q1, Q2, Q3, Q4) and check distribution
        
        WHY THIS MATTERS:
        ----------------
        If 80% of data from 2019-2020 (pre-COVID) but production is 2024,
        model won't understand current customer behavior
        """
        # Extract year-quarter from timestamps
        def extract_quarter(ts):
            try:
                dt = pd.to_datetime(ts)
                return f"{dt.year}-Q{(dt.month-1)//3 + 1}"
            except:
                return 'invalid'
        
        train_quarters = self.train_df['timestamp'].apply(extract_quarter)
        prod_quarters = self.prod_df['timestamp'].apply(extract_quarter)
        
        # Create temporary column for KL calculation
        train_temp = pd.DataFrame({'quarter': train_quarters})
        prod_temp = pd.DataFrame({'quarter': prod_quarters})
        
        # Compute KL on quarters
        prod_counts = prod_temp['quarter'].value_counts(normalize=True, dropna=True)
        train_counts = train_temp['quarter'].value_counts(normalize=True, dropna=True)
        
        all_quarters = set(prod_counts.index) | set(train_counts.index)
        epsilon = 1e-10
        
        prod_probs = np.array([prod_counts.get(q, epsilon) for q in all_quarters])
        train_probs = np.array([train_counts.get(q, epsilon) for q in all_quarters])
        
        prod_probs = prod_probs / prod_probs.sum()
        train_probs = train_probs / train_probs.sum()
        
        kl = np.sum(prod_probs * np.log(prod_probs / train_probs))
        
        self.metrics['temporal_consistency'] = {
            'kl_divergence': float(kl),
            'threshold': 0.1,
            'pass': bool(kl < 0.1)
        }
        
        return float(kl)
    
    def get_all_metrics(self) -> Dict:
        """
        Run all DDQ assessments.
        
        Returns:
            Dictionary with all DDQ metrics
        """
        self.assess_category_balance()
        self.assess_sentiment_distribution()
        self.assess_temporal_consistency()
        
        return self.metrics


class DataCleaner:
    """
    Applies data cleaning strategies to fix quality issues.
    
    CLEANING STRATEGIES:
    -------------------
    1. Remove duplicates
    2. Handle missing values (drop or impute)
    3. Fix invalid formats
    4. Rebalance distributions (optional)
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: DataFrame to clean
        """
        self.df = df.copy()
        self.cleaning_log = []
    
    def remove_duplicates(self) -> pd.DataFrame:
        """
        Remove duplicate records.
        
        STRATEGY:
        --------
        Keep first occurrence, drop subsequent duplicates
        """
        before_count = len(self.df)
        self.df = self.df.drop_duplicates()
        after_count = len(self.df)
        
        self.cleaning_log.append({
            'operation': 'remove_duplicates',
            'removed': before_count - after_count
        })
        
        return self.df
    
    def handle_missing_values(self, strategy: str = 'drop') -> pd.DataFrame:
        """
        Handle missing values.
        
        STRATEGIES:
        ----------
        - 'drop': Remove rows with any NaN (conservative)
        - 'impute': Fill with mode/median (aggressive)
        
        Args:
            strategy: 'drop' or 'impute'
        """
        before_count = len(self.df)
        
        if strategy == 'drop':
            self.df = self.df.dropna()
        elif strategy == 'impute':
            # Fill categorical with mode, numerical with median
            for col in self.df.columns:
                if self.df[col].dtype == 'object':
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                else:
                    self.df[col].fillna(self.df[col].median(), inplace=True)
        
        after_count = len(self.df)
        
        self.cleaning_log.append({
            'operation': f'handle_missing_{strategy}',
            'removed': before_count - after_count
        })
        
        return self.df
    
    def fix_invalid_timestamps(self) -> pd.DataFrame:
        """
        Remove rows with invalid timestamp formats.
        
        APPROACH:
        --------
        Try parsing each timestamp; if fails, mark for removal
        """
        before_count = len(self.df)
        
        if 'timestamp' in self.df.columns:
            # Keep only valid ISO timestamps
            iso_pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}'
            valid_mask = self.df['timestamp'].astype(str).str.match(iso_pattern)
            self.df = self.df[valid_mask]
        
        after_count = len(self.df)
        
        self.cleaning_log.append({
            'operation': 'fix_invalid_timestamps',
            'removed': before_count - after_count
        })
        
        return self.df
    
    def get_cleaned_data(self) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Apply all cleaning operations in sequence.
        
        Returns:
            (cleaned_df, cleaning_log)
        """
        self.remove_duplicates()
        self.fix_invalid_timestamps()
        self.handle_missing_values(strategy='drop')
        
        return self.df, self.cleaning_log


def generate_production_distribution(n_records: int = 10000) -> pd.DataFrame:
    """
    Generate "ideal" production distribution for DDQ comparison.
    
    PRODUCTION CHARACTERISTICS:
    --------------------------
    - Balanced categories (25% each instead of 60/30/7/3)
    - Balanced sentiment (33% each)
    - Even temporal spread (uniform across 5 years)
    - No quality issues (100% complete, valid, unique)
    
    This represents the "real-world" distribution we expect in production.
    
    Args:
        n_records: Number of production records
        
    Returns:
        Production DataFrame
    """
    np.random.seed(123)  # Different seed than training
    
    # BALANCED distributions (ideal production)
    categories = ['Electronics', 'Clothing', 'Home', 'Sports']
    category_weights = [0.25, 0.25, 0.25, 0.25]  # Balanced!
    
    sentiments = ['positive', 'neutral', 'negative']
    sentiment_weights = [0.33, 0.34, 0.33]  # Balanced!
    
    segments = ['new', 'regular', 'vip']
    segment_weights = [0.4, 0.4, 0.2]
    
    outcomes = ['purchase', 'abandon', 'escalate']
    outcome_weights = [0.35, 0.55, 0.1]
    
    data = []
    
    for i in range(n_records):
        # Generate timestamp (uniform across 5 years)
        start_date = datetime(2019, 1, 1)
        end_date = datetime(2024, 12, 31)
        time_between = end_date - start_date
        random_days = np.random.randint(0, time_between.days)
        timestamp = start_date + timedelta(days=random_days)
        
        record = {
            'customer_id': f'PROD_{i:06d}',
            'product_category': np.random.choice(categories, p=category_weights),
            'sentiment': np.random.choice(sentiments, p=sentiment_weights),
            'timestamp': timestamp.isoformat(),
            'conversation_text': 'Production conversation text',
            'outcome': np.random.choice(outcomes, p=outcome_weights)
        }
        
        data.append(record)
    
    return pd.DataFrame(data)


def generate_report(dq_metrics: Dict, ddq_metrics: Dict, 
                   cleaning_log: List[Dict]) -> Dict:
    """
    Generate comprehensive JSON report.
    
    REPORT STRUCTURE:
    ----------------
    {
        "dq_assessment": {...},
        "ddq_assessment": {...},
        "cleaning_summary": {...},
        "overall_status": "PASS" or "FAIL"
    }
    
    Args:
        dq_metrics: Data Quality metrics
        ddq_metrics: Distributional Quality metrics
        cleaning_log: List of cleaning operations
        
    Returns:
        Complete report dictionary
    """
    # Check if all metrics pass
    dq_pass = all(metric.get('pass', False) for metric in dq_metrics.values())
    ddq_pass = all(metric.get('pass', False) for metric in ddq_metrics.values())
    
    overall_pass = dq_pass and ddq_pass
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "dq_assessment": {
            "metrics": dq_metrics,
            "status": "PASS" if dq_pass else "FAIL"
        },
        "ddq_assessment": {
            "metrics": ddq_metrics,
            "status": "PASS" if ddq_pass else "FAIL"
        },
        "cleaning_summary": {
            "operations": cleaning_log,
            "total_operations": len(cleaning_log)
        },
        "overall_status": "PASS" if overall_pass else "FAIL"
    }
    
    return report


def main():
    """
    Main execution pipeline for Part 1.
    
    WORKFLOW:
    --------
    1. Generate synthetic training data (with quality issues)
    2. Generate production data (ideal distribution)
    3. Assess DQ metrics
    4. Assess DDQ metrics
    5. Clean data
    6. Re-assess after cleaning
    7. Generate report
    """
    print("=" * 80)
    print("PART 1: DATA QUALITY (DQ) & DISTRIBUTIONAL QUALITY (DDQ)")
    print("=" * 80)
    
    # Step 1: Generate data
    print("\n[Step 1] Generating synthetic retail conversation data...")
    generator = RetailDataGenerator(n_records=10000, seed=42)
    raw_data = generator.generate_data()
    print(f"✓ Generated {len(raw_data)} records")
    print(f"  Columns: {list(raw_data.columns)}")
    
    # Step 2: Generate production data
    print("\n[Step 2] Generating production distribution for comparison...")
    prod_data = generate_production_distribution(n_records=10000)
    print(f"✓ Generated {len(prod_data)} production records")
    
    # Step 3: Assess DQ (before cleaning)
    print("\n[Step 3] Assessing Data Quality (DQ) metrics...")
    dq_assessor = DataQualityAssessor(raw_data)
    dq_metrics = dq_assessor.get_all_metrics()
    
    print(f"  Completeness: {dq_metrics['completeness']['value']:.4f} "
          f"(Threshold: ≥ {dq_metrics['completeness']['threshold']})")
    print(f"  Duplicate Rate: {dq_metrics['duplicate_rate']['value']:.4f} "
          f"(Threshold: ≥ {dq_metrics['duplicate_rate']['threshold']})")
    print(f"  Format Validity: {dq_metrics['format_validity']['value']:.4f} "
          f"(Threshold: ≥ {dq_metrics['format_validity']['threshold']})")
    
    # Step 4: Assess DDQ (before cleaning)
    print("\n[Step 4] Assessing Distributional Data Quality (DDQ) metrics...")
    ddq_assessor = DistributionalQualityAssessor(raw_data, prod_data)
    ddq_metrics = ddq_assessor.get_all_metrics()
    
    print(f"  Category Balance KL: {ddq_metrics['category_balance']['kl_divergence']:.4f} "
          f"(Threshold: < {ddq_metrics['category_balance']['threshold']})")
    print(f"  Sentiment Distribution KL: {ddq_metrics['sentiment_distribution']['kl_divergence']:.4f} "
          f"(Threshold: < {ddq_metrics['sentiment_distribution']['threshold']})")
    print(f"  Temporal Consistency KL: {ddq_metrics['temporal_consistency']['kl_divergence']:.4f} "
          f"(Threshold: < {ddq_metrics['temporal_consistency']['threshold']})")
    
    # Step 5: Clean data
    print("\n[Step 5] Cleaning data...")
    cleaner = DataCleaner(raw_data)
    cleaned_data, cleaning_log = cleaner.get_cleaned_data()
    print(f"✓ Cleaned data: {len(raw_data)} → {len(cleaned_data)} records")
    for log in cleaning_log:
        print(f"  - {log['operation']}: removed {log['removed']} records")
    
    # Step 6: Re-assess after cleaning
    print("\n[Step 6] Re-assessing quality after cleaning...")
    dq_assessor_clean = DataQualityAssessor(cleaned_data)
    dq_metrics_clean = dq_assessor_clean.get_all_metrics()
    
    print(f"  Completeness (after): {dq_metrics_clean['completeness']['value']:.4f} ✓")
    print(f"  Duplicate Rate (after): {dq_metrics_clean['duplicate_rate']['value']:.4f} ✓")
    print(f"  Format Validity (after): {dq_metrics_clean['format_validity']['value']:.4f} ✓")
    
    # Step 7: Generate report
    print("\n[Step 7] Generating quality report...")
    report = generate_report(dq_metrics_clean, ddq_metrics, cleaning_log)
    
    # Save report
    report_path = '/Users/mohitpansari/Downloads/workspace/RL_Assignment/quality_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"✓ Report saved to: {report_path}")
    
    # Save cleaned data
    cleaned_data.to_csv('/Users/mohitpansari/Downloads/workspace/RL_Assignment/cleaned_data.csv', 
                        index=False)
    print(f"✓ Cleaned data saved to: cleaned_data.csv")
    
    # Final status
    print("\n" + "=" * 80)
    print(f"OVERALL STATUS: {report['overall_status']}")
    print("=" * 80)
    
    return cleaned_data, report


if __name__ == "__main__":
    cleaned_data, report = main()