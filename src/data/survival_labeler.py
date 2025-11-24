"""
Survival Analysis Labeling Module

This module handles labeling for survival analysis, creating:
1. Structured arrays for scikit-survival (event, duration)
2. Categorical labels for analysis and comparison
3. Statistics and validation

Author: Claude Code
Date: 2025-11-14
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SurvivalLabeler:
    """
    Creates survival labels for coffee shop data based on date_created and date_closed.

    Survival Analysis Format:
        - event: True if closed, False if still operating (censored)
        - duration: Days from date_created to date_closed (or to reference_date if still open)

    Categorical Labels (for analysis):
        - 0: Failed (closed before threshold_days)
        - 1: Success (operating >= threshold_days OR closed after >= threshold_days)
        - 2: Too new (still open but < threshold_days old)
    """

    def __init__(self, threshold_days: int = 730, reference_date: datetime = None):
        """
        Initialize survival labeler.

        Args:
            threshold_days: Survival threshold in days (default: 730 = 2 years)
            reference_date: Reference date for calculating duration of open shops
                          (default: today)
        """
        self.threshold_days = threshold_days
        self.reference_date = reference_date or pd.Timestamp.now()

        logger.info(f"SurvivalLabeler initialized:")
        logger.info(f"  - Threshold: {threshold_days} days ({threshold_days/365:.1f} years)")
        logger.info(f"  - Reference date: {self.reference_date.date()}")

    def create_survival_labels(
        self,
        df: pd.DataFrame,
        date_created_col: str = 'date_created',
        date_closed_col: str = 'date_closed'
    ) -> pd.DataFrame:
        """
        Create survival analysis labels from date columns.

        Args:
            df: DataFrame with coffee shop data
            date_created_col: Column name for creation date
            date_closed_col: Column name for closure date (NULL if still open)

        Returns:
            DataFrame with added columns:
                - event: Boolean (True=closed, False=censored/still open)
                - duration: Float (days from creation to closure or reference_date)
                - categorical_label: Int (0=Failed, 1=Success, 2=Too new)
        """
        logger.info(f"Creating survival labels for {len(df)} coffee shops...")

        # Make a copy to avoid modifying original
        result = df.copy()

        # Parse dates
        logger.info("Parsing dates...")
        result['date_created_parsed'] = pd.to_datetime(
            result[date_created_col], errors='coerce'
        )
        result['date_closed_parsed'] = pd.to_datetime(
            result[date_closed_col], errors='coerce'
        )

        # Check for invalid dates
        invalid_created = result['date_created_parsed'].isna().sum()
        if invalid_created > 0:
            logger.warning(f"Found {invalid_created} shops with invalid date_created")
            result = result[result['date_created_parsed'].notna()].copy()

        # Create event indicator
        # event = True if date_closed is NOT NULL (shop closed)
        # event = False if date_closed is NULL (shop still operating - censored observation)
        result['event'] = result['date_closed_parsed'].notna()

        logger.info(f"Event distribution:")
        logger.info(f"  - Closed (event=True): {result['event'].sum()}")
        logger.info(f"  - Still open (event=False): {(~result['event']).sum()}")

        # Calculate duration
        # For closed shops: duration = date_closed - date_created
        # For open shops: duration = reference_date - date_created (right-censored)
        result['duration'] = np.where(
            result['event'],
            (result['date_closed_parsed'] - result['date_created_parsed']).dt.days,
            (self.reference_date - result['date_created_parsed']).dt.days
        )

        # Handle negative or zero durations (data quality issues)
        negative_durations = (result['duration'] <= 0).sum()
        if negative_durations > 0:
            logger.warning(f"Found {negative_durations} shops with duration <= 0, setting to 1 day")
            result.loc[result['duration'] <= 0, 'duration'] = 1.0

        # Log duration statistics
        logger.info(f"Duration statistics (days):")
        logger.info(f"  - Mean: {result['duration'].mean():.0f} ({result['duration'].mean()/365:.1f} years)")
        logger.info(f"  - Median: {result['duration'].median():.0f} ({result['duration'].median()/365:.1f} years)")
        logger.info(f"  - Min: {result['duration'].min():.0f}")
        logger.info(f"  - Max: {result['duration'].max():.0f} ({result['duration'].max()/365:.1f} years)")

        # Create categorical labels for analysis
        result['categorical_label'] = self._create_categorical_labels(result)

        # Log label distribution
        label_counts = result['categorical_label'].value_counts().sort_index()
        logger.info(f"\nCategorical label distribution (threshold={self.threshold_days} days):")
        logger.info(f"  - Label 0 (Failed <{self.threshold_days/365:.1f}yr): {label_counts.get(0, 0)} ({label_counts.get(0, 0)/len(result)*100:.1f}%)")
        logger.info(f"  - Label 1 (Success >={self.threshold_days/365:.1f}yr): {label_counts.get(1, 0)} ({label_counts.get(1, 0)/len(result)*100:.1f}%)")
        logger.info(f"  - Label 2 (Too new): {label_counts.get(2, 0)} ({label_counts.get(2, 0)/len(result)*100:.1f}%)")

        return result

    def _create_categorical_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Create categorical labels based on survival threshold.

        Label 0 (Failed): Closed AND duration < threshold_days
        Label 1 (Success): Duration >= threshold_days (regardless of closure status)
        Label 2 (Too new): Still open AND duration < threshold_days

        Args:
            df: DataFrame with 'event' and 'duration' columns

        Returns:
            Series with categorical labels (0, 1, or 2)
        """
        labels = pd.Series(index=df.index, dtype=int)

        # Label 0: Failed (closed before threshold)
        failed_mask = (df['event'] == True) & (df['duration'] < self.threshold_days)
        labels[failed_mask] = 0

        # Label 1: Success (survived past threshold, may or may not have closed later)
        success_mask = (df['duration'] >= self.threshold_days)
        labels[success_mask] = 1

        # Label 2: Too new (still open but haven't reached threshold yet)
        too_new_mask = (df['event'] == False) & (df['duration'] < self.threshold_days)
        labels[too_new_mask] = 2

        return labels

    def create_structured_array(self, df: pd.DataFrame) -> np.ndarray:
        """
        Create structured numpy array for scikit-survival models.

        The structured array has dtype=[('event', bool), ('duration', float)]
        which is required by scikit-survival's fit() method.

        Args:
            df: DataFrame with 'event' and 'duration' columns

        Returns:
            Structured numpy array with event and duration
        """
        # Create structured array as required by scikit-survival
        y = np.array(
            list(zip(df['event'], df['duration'])),
            dtype=[('event', bool), ('duration', float)]
        )

        logger.info(f"Created structured array: shape={y.shape}, dtype={y.dtype}")
        return y

    def get_label_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Get comprehensive statistics about the labels.

        Args:
            df: DataFrame with survival labels

        Returns:
            Dictionary with label statistics
        """
        stats = {
            'total_shops': len(df),
            'threshold_days': self.threshold_days,
            'threshold_years': self.threshold_days / 365,
            'reference_date': str(self.reference_date.date()),

            # Event statistics
            'closed_count': int(df['event'].sum()),
            'closed_pct': float(df['event'].sum() / len(df) * 100),
            'censored_count': int((~df['event']).sum()),
            'censored_pct': float((~df['event']).sum() / len(df) * 100),

            # Duration statistics
            'duration_mean_days': float(df['duration'].mean()),
            'duration_mean_years': float(df['duration'].mean() / 365),
            'duration_median_days': float(df['duration'].median()),
            'duration_median_years': float(df['duration'].median() / 365),
            'duration_min_days': float(df['duration'].min()),
            'duration_max_days': float(df['duration'].max()),
            'duration_max_years': float(df['duration'].max() / 365),

            # Categorical label statistics
            'label_0_count': int((df['categorical_label'] == 0).sum()),
            'label_0_pct': float((df['categorical_label'] == 0).sum() / len(df) * 100),
            'label_1_count': int((df['categorical_label'] == 1).sum()),
            'label_1_pct': float((df['categorical_label'] == 1).sum() / len(df) * 100),
            'label_2_count': int((df['categorical_label'] == 2).sum()),
            'label_2_pct': float((df['categorical_label'] == 2).sum() / len(df) * 100),
        }

        # Closed shops duration statistics
        closed_shops = df[df['event'] == True]
        if len(closed_shops) > 0:
            stats.update({
                'closed_duration_mean_days': float(closed_shops['duration'].mean()),
                'closed_duration_mean_years': float(closed_shops['duration'].mean() / 365),
                'closed_duration_median_days': float(closed_shops['duration'].median()),
                'closed_duration_median_years': float(closed_shops['duration'].median() / 365),
            })

        # Still open shops age statistics
        open_shops = df[df['event'] == False]
        if len(open_shops) > 0:
            stats.update({
                'open_age_mean_days': float(open_shops['duration'].mean()),
                'open_age_mean_years': float(open_shops['duration'].mean() / 365),
                'open_age_median_days': float(open_shops['duration'].median()),
                'open_age_median_years': float(open_shops['duration'].median() / 365),
            })

        return stats

    def validate_labels(self, df: pd.DataFrame) -> Tuple[bool, list]:
        """
        Validate that labels are correctly created.

        Args:
            df: DataFrame with survival labels

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Check required columns
        required_cols = ['event', 'duration', 'categorical_label']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
            return False, issues

        # Check for NaN values
        for col in required_cols:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                issues.append(f"Found {nan_count} NaN values in {col}")

        # Check duration values
        if (df['duration'] <= 0).any():
            issues.append(f"Found {(df['duration'] <= 0).sum()} shops with duration <= 0")

        # Check categorical labels are in valid range
        invalid_labels = df[~df['categorical_label'].isin([0, 1, 2])]
        if len(invalid_labels) > 0:
            issues.append(f"Found {len(invalid_labels)} shops with invalid categorical labels")

        # Logical checks
        # Label 0 should be: event=True AND duration < threshold
        label_0 = df[df['categorical_label'] == 0]
        if len(label_0) > 0:
            wrong_event = label_0[label_0['event'] != True]
            if len(wrong_event) > 0:
                issues.append(f"Found {len(wrong_event)} label 0 shops with event != True")

            wrong_duration = label_0[label_0['duration'] >= self.threshold_days]
            if len(wrong_duration) > 0:
                issues.append(f"Found {len(wrong_duration)} label 0 shops with duration >= threshold")

        # Label 2 should be: event=False AND duration < threshold
        label_2 = df[df['categorical_label'] == 2]
        if len(label_2) > 0:
            wrong_event = label_2[label_2['event'] != False]
            if len(wrong_event) > 0:
                issues.append(f"Found {len(wrong_event)} label 2 shops with event != False")

            wrong_duration = label_2[label_2['duration'] >= self.threshold_days]
            if len(wrong_duration) > 0:
                issues.append(f"Found {len(wrong_duration)} label 2 shops with duration >= threshold")

        is_valid = len(issues) == 0

        if is_valid:
            logger.info("Label validation passed!")
        else:
            logger.warning(f"Label validation found {len(issues)} issues:")
            for issue in issues:
                logger.warning(f"  - {issue}")

        return is_valid, issues


def create_labels_for_multiple_thresholds(
    df: pd.DataFrame,
    thresholds_days: list = [180, 365, 730, 1095],
    date_created_col: str = 'date_created',
    date_closed_col: str = 'date_closed'
) -> Dict[int, pd.DataFrame]:
    """
    Create survival labels for multiple time thresholds.

    This is useful for comparing model performance at different time points
    (e.g., 6 months, 1 year, 2 years, 3 years).

    Args:
        df: DataFrame with coffee shop data
        thresholds_days: List of thresholds in days (default: [180, 365, 730, 1095])
        date_created_col: Column name for creation date
        date_closed_col: Column name for closure date

    Returns:
        Dictionary mapping threshold_days to labeled DataFrames
    """
    results = {}

    logger.info(f"Creating labels for {len(thresholds_days)} thresholds: {thresholds_days}")

    for threshold in thresholds_days:
        logger.info(f"\n--- Processing threshold: {threshold} days ({threshold/365:.1f} years) ---")
        labeler = SurvivalLabeler(threshold_days=threshold)
        labeled_df = labeler.create_survival_labels(
            df,
            date_created_col=date_created_col,
            date_closed_col=date_closed_col
        )
        results[threshold] = labeled_df

    return results
