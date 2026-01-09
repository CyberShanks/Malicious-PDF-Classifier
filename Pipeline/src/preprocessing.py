"""
PDF Preprocessing - Model3 Approach
Winsorization + log1p + StandardScaler to handle outliers and skewness.
Keeps training and prediction preprocessing identical.
"""

import logging
from typing import Dict, List, Optional, Union
import polars as pl
import numpy as np
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class PDFTransformer:
    """
    Applies Winsorization (cap outliers) + log1p transform (fix skewness).

    Must be fitted on training data to learn the 99th percentile caps.
    Then applies same caps at prediction time.
    """

    # These columns are all zeros in the dataset - useless
    COLUMNS_TO_DROP_ALWAYS = [
        "embedded_file_count",
        "average_embedded_file_size",
        "xref_count",
        "xref_entries",
        "submitform_count",
        "jbig2decode_count",
        "trailer_count",
        "startxref_count"
    ]

    # Drop these after initial cleaning
    COLUMNS_TO_DROP_AFTER = ["file_path"]

    def __init__(self, caps: Optional[Dict[str, float]] = None):
        """
        Initialize transformer.

        Args:
            caps: Pre-computed 99th percentile caps (for prediction mode).
                  If None, must call fit() first (training mode).
        """
        self.caps = caps
        self.numeric_cols = None

    def fit(self, df: pl.DataFrame) -> 'PDFTransformer':
        """
        Learn 99th percentile caps from training data.

        Args:
            df: Training DataFrame with raw features

        Returns:
            self (for chaining)
        """
        # Drop always-empty columns
        cols_to_drop = [c for c in self.COLUMNS_TO_DROP_ALWAYS if c in df.columns]
        if cols_to_drop:
            df = df.drop(cols_to_drop)

        # Get numeric columns (exclude label)
        self.numeric_cols = df.select(pl.col(pl.Int64, pl.Float64)).columns
        if "label" in self.numeric_cols:
            self.numeric_cols.remove("label")

        # Calculate 99th percentile for each numeric column
        logger.info(f"Fitting transformer on {len(self.numeric_cols)} features...")
        caps_df = df.select([
            pl.col(c).quantile(0.99).alias(c) for c in self.numeric_cols
        ])
        self.caps = caps_df.to_dict(as_series=False)

        # Convert from dict of lists to dict of scalars
        self.caps = {k: v[0] for k, v in self.caps.items()}

        logger.info(f"  ✅ Learned caps for {len(self.caps)} features")
        return self

    def transform(self, features: Dict[str, any]) -> Dict[str, any]:
        """
        Apply Winsorization + log1p to a single sample (for prediction).

        Args:
            features: Raw PDF features from PDFFeatureExtractor

        Returns:
            Transformed features
        """
        if self.caps is None:
            raise ValueError("Transformer not fitted! Call fit() first or provide caps in __init__")

        transformed = features.copy()

        # Drop columns we don't need
        for col in self.COLUMNS_TO_DROP_ALWAYS + self.COLUMNS_TO_DROP_AFTER:
            transformed.pop(col, None)

        # Apply Winsorization (cap at 99th percentile) + log1p
        for col, cap_val in self.caps.items():
            if col in transformed:
                # Cap the value
                transformed[col] = min(transformed[col], cap_val)
                # Apply log1p
                transformed[col] = np.log1p(transformed[col])

        return transformed

    def transform_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply Winsorization + log1p to a DataFrame (for batch processing).

        Args:
            df: DataFrame with raw features

        Returns:
            Transformed DataFrame
        """
        if self.caps is None:
            raise ValueError("Transformer not fitted! Call fit() first or provide caps in __init__")

        # Drop always-empty columns
        cols_to_drop = [c for c in self.COLUMNS_TO_DROP_ALWAYS if c in df.columns]
        if cols_to_drop:
            df = df.drop(cols_to_drop)

        # Drop file_path if present
        if "file_path" in df.columns:
            df = df.drop("file_path")

        # Apply Winsorization + log1p
        ops = []
        for col in self.numeric_cols:
            if col in df.columns:
                cap_val = self.caps[col]
                ops.append(
                    pl.col(col)
                    .clip(upper_bound=cap_val)  # Winsorization
                    .log1p()                     # Fix skewness
                    .alias(col)
                )

        df = df.with_columns(ops)
        return df

    def get_caps(self) -> Dict[str, float]:
        """Get the fitted caps (for saving to metadata)."""
        if self.caps is None:
            raise ValueError("Transformer not fitted!")
        return self.caps.copy()


class PDFFeatureSelector:
    """
    Drops sparse features (>99% zeros) that don't help the model.
    Based on analysis in Model3.ipynb.
    """

    # Features to remove (from Model3.ipynb analysis)
    SPARSE_REMOVED = {
        "encrypted",
        "uses_nonstandard_port",
        "launch_count",
        "richmedia_count"
    }

    # What's left after filtering (28 features from Model3)
    SELECTED_FEATURES = [
        "file_size",
        "title_chars",
        "metadata_size",
        "page_count",
        "valid_pdf_header",
        "image_count",
        "text_length",
        "object_count",
        "font_object_count",
        "stream_count",
        "endstream_count",
        "average_stream_size",
        "entropy_of_streams",
        "name_obfuscations",
        "total_filters",
        "nested_filter_objects",
        "objstm_count",
        "js_count",
        "javascript_count",
        "uri_count",
        "action_count",
        "aa_count",
        "openaction_count",
        "acroform_count",
        "xfa_count",
        "colors_count",
        "has_multiple_behavioral_keywords_in_one_object",
        "used_ocr"
    ]

    @classmethod
    def select_features(cls, features: Dict[str, any]) -> Dict[str, any]:
        """
        Select only the features used by the model (single sample).

        Args:
            features: Dictionary with all features (after transformation)

        Returns:
            Dictionary with only selected features
        """
        # Remove sparse features
        filtered = {
            k: v for k, v in features.items()
            if k not in cls.SPARSE_REMOVED
        }

        # Keep only selected features (in correct order)
        selected = {k: filtered.get(k, 0) for k in cls.SELECTED_FEATURES}

        return selected

    @classmethod
    def select_features_dataframe(cls, df: pl.DataFrame) -> pl.DataFrame:
        """
        Select only the features used by the model (for DataFrames).

        Args:
            df: DataFrame with all features

        Returns:
            DataFrame with only selected features (and label if present)
        """
        # Get columns to keep
        cols_to_keep = [c for c in cls.SELECTED_FEATURES if c in df.columns]

        # Add label if it exists
        if "label" in df.columns:
            cols_to_keep.append("label")

        return df.select(cols_to_keep)


class PDFPreprocessingPipeline:
    """
    Complete preprocessing pipeline: Winsorization + log1p + scaling + feature selection.

    This ensures identical preprocessing for training and prediction.

    Example:
        # Training mode
        >>> pipeline = PDFPreprocessingPipeline()
        >>> pipeline.fit(train_df)
        >>> train_processed = pipeline.transform_dataframe(train_df)
        >>> caps = pipeline.get_caps()  # Save these!
        >>> scaler = pipeline.get_scaler()  # Save this too!

        # Prediction mode
        >>> pipeline = PDFPreprocessingPipeline(caps=caps, scaler=scaler)
        >>> processed = pipeline.transform(raw_features)
    """

    def __init__(self, caps: Optional[Dict[str, float]] = None,
                 scaler: Optional[StandardScaler] = None,
                 n_features: Optional[int] = None):
        """
        Initialize pipeline.

        Args:
            caps: Pre-computed transformation caps (for prediction mode).
                  If None, must call fit() first (training mode).
            scaler: Pre-fitted StandardScaler (for prediction mode).
                    If None, must call fit() first (training mode).
            n_features: Expected number of features for the model.
                       If None, uses default (28 features with sparse removal).
                       If 32, skips sparse feature removal (keeps all 32 features).
        """
        self.transformer = PDFTransformer(caps=caps)
        self.selector = PDFFeatureSelector()
        self.scaler = scaler
        self.n_features = n_features
        self.skip_feature_selection = (n_features == 32) if n_features else False

    def fit(self, df: pl.DataFrame) -> 'PDFPreprocessingPipeline':
        """
        Fit the pipeline on training data.

        Args:
            df: Training DataFrame with raw features

        Returns:
            self (for chaining)
        """
        # Fit transformer (learn caps)
        self.transformer.fit(df)

        # Transform and optionally select features
        transformed = self.transformer.transform_dataframe(df)

        if self.skip_feature_selection:
            # Keep all 32 features (no sparse removal)
            logger.info(f"  ℹ️  Skipping sparse feature removal (keeping all features for 32-feature model)")
            selected = transformed
        else:
            # Apply sparse removal (28 features)
            selected = self.selector.select_features_dataframe(transformed)

        # Fit scaler on transformed features
        X = selected.drop("label").to_numpy() if "label" in selected.columns else selected.to_numpy()
        self.scaler = StandardScaler()
        self.scaler.fit(X)

        logger.info(f"  ✅ Scaler fitted on {X.shape[1]} features")

        return self

    def transform(self, features: Dict[str, any]) -> np.ndarray:
        """
        Complete preprocessing: transformation + selection + scaling.

        Args:
            features: Raw PDF features from PDFFeatureExtractor

        Returns:
            Scaled numpy array ready for model prediction (1, n_features)
        """
        if self.scaler is None:
            raise ValueError("Pipeline not fitted! Call fit() first or provide scaler in __init__")

        # Step 1: Winsorization + log1p
        transformed = self.transformer.transform(features)

        # Step 2: Feature selection (conditional)
        if self.skip_feature_selection:
            # Keep all 32 features (no sparse removal)
            selected = transformed
            # Get feature names from transformer (all numeric features)
            feature_names = sorted(selected.keys())
        else:
            # Apply sparse removal (28 features)
            selected = self.selector.select_features(transformed)
            feature_names = self.selector.SELECTED_FEATURES

        # Step 3: Convert to array in correct order
        X = np.array([[selected.get(f, 0) for f in feature_names]])

        # Step 4: Standardize
        X_scaled = self.scaler.transform(X)

        return X_scaled

    def transform_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Complete preprocessing for DataFrames.

        Args:
            df: DataFrame with raw features

        Returns:
            Processed DataFrame ready for model training
        """
        # Step 1: Winsorization + log1p
        transformed = self.transformer.transform_dataframe(df)

        # Step 2: Feature selection (conditional)
        if self.skip_feature_selection:
            # Keep all 32 features
            selected = transformed
        else:
            # Apply sparse removal (28 features)
            selected = self.selector.select_features_dataframe(transformed)

        return selected

    def get_feature_names(self) -> List[str]:
        """Get the final list of feature names in correct order."""
        if self.skip_feature_selection:
            # Return all numeric features (32 features)
            if self.transformer.numeric_cols:
                return sorted(self.transformer.numeric_cols)
            else:
                # Fallback if not fitted yet
                return []
        else:
            # Return selected features (28 features)
            return self.selector.SELECTED_FEATURES.copy()

    def get_caps(self) -> Dict[str, float]:
        """Get the fitted transformation caps (for saving to metadata)."""
        return self.transformer.get_caps()

    def get_scaler(self) -> StandardScaler:
        """Get the fitted scaler (for saving to model directory)."""
        if self.scaler is None:
            raise ValueError("Scaler not fitted! Call fit() first.")
        return self.scaler


# Convenience function
def preprocess_pdf_features(features: Dict[str, any], caps: Dict[str, float],
                            scaler: StandardScaler) -> np.ndarray:
    """
    Convenience function to preprocess raw PDF features.

    Args:
        features: Raw features from PDFFeatureExtractor
        caps: Transformation caps from training
        scaler: Fitted StandardScaler from training

    Returns:
        Scaled numpy array ready for model
    """
    pipeline = PDFPreprocessingPipeline(caps=caps, scaler=scaler)
    return pipeline.transform(features)
