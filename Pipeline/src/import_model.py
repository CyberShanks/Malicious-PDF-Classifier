"""
Model Import

Loads a trained model and checks if it works with our pipeline.

Usage:
    python src/import_model.py model.joblib --test
"""

import sys
import json
import pickle
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional

import joblib
import polars as pl
import numpy as np

# Import from same directory
sys.path.insert(0, str(Path(__file__).parent))
from preprocessing import PDFPreprocessingPipeline
from features import PDFFeatureExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelImporter:
    """Loads a model and checks compatibility with our preprocessing."""

    def __init__(self):
        self.model_dir = Path("models")
        self.pipeline = None  # Will be fitted during import

    def load_model_file(self, model_path: str):
        """Try joblib first, then pickle as fallback."""
        logger.info(f"Loading model from {model_path}...")

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            model = joblib.load(model_path)
            logger.info("  ✅ Loaded with joblib")
            return model
        except Exception as e1:
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                logger.info("  ✅ Loaded with pickle")
                return model
            except Exception as e2:
                raise Exception(f"Failed to load model. Joblib error: {e1}, Pickle error: {e2}")

    def load_pickle(self, pickle_path: str):
        """Old name - just calls load_model_file."""
        return self.load_model_file(pickle_path)

    def extract_model_from_dict(self, obj):
        """
        Extract actual model from various dict formats.

        Handles common patterns:
        - {'model': actual_model}
        - {'best_model': actual_model}
        - {'estimator': actual_model}
        - {'clf': actual_model}
        - {'classifier': actual_model}
        - {'pipeline': actual_model}
        - Multiple models: returns the first one found

        Args:
            obj: Loaded object (might be dict, model, or something else)

        Returns:
            Extracted model or original object
        """
        if not isinstance(obj, dict):
            return obj

        logger.info("  📦 Detected dict wrapper, extracting model...")

        # Common keys for models
        model_keys = [
            'model', 'best_model', 'estimator', 'clf', 'classifier',
            'pipeline', 'trained_model', 'final_model'
        ]

        # Try common keys first
        for key in model_keys:
            if key in obj:
                logger.info(f"  ✅ Found model at key: '{key}'")
                return obj[key]

        # Look for anything with 'predict' method (likely a model)
        for key, value in obj.items():
            if hasattr(value, 'predict'):
                logger.info(f"  ✅ Found model-like object at key: '{key}'")
                return value

        # If dict has sklearn Pipeline attributes, it might be the model itself
        if hasattr(obj, 'predict'):
            logger.info("  ✅ Dict has predict method, using as-is")
            return obj

        # Last resort: show what's in the dict
        logger.warning("  ⚠️  Could not find model in dict!")
        logger.warning(f"  Available keys: {list(obj.keys())}")

        # Try to be helpful
        if len(obj) == 1:
            only_key = list(obj.keys())[0]
            only_value = obj[only_key]
            if hasattr(only_value, 'predict'):
                logger.info(f"  ✅ Using only value at '{only_key}'")
                return only_value

        raise ValueError(
            f"Could not extract model from dict. Available keys: {list(obj.keys())}. "
            f"Please save model directly or use one of these keys: {model_keys}"
        )

    def extract_classifier_from_pipeline(self, model):
        """
        If model is a Pipeline with built-in scaler, extract just the classifier.
        This ensures we can use our own preprocessing consistently.

        Args:
            model: Model (might be Pipeline or standalone classifier)

        Returns:
            Classifier (with note about extraction if applicable)
        """
        if not hasattr(model, 'named_steps'):
            return model  # Not a pipeline, return as-is

        steps = model.named_steps

        # Check if pipeline has a scaler
        has_scaler = any(
            'scaler' in name.lower() or 'standard' in name.lower() or 'robust' in name.lower()
            for name in steps.keys()
        )

        if not has_scaler:
            return model  # No scaler, can use pipeline as-is

        # Extract the last step (usually the classifier)
        last_step_name = list(steps.keys())[-1]
        classifier = steps[last_step_name]

        logger.info(f"  🔧 Extracting classifier '{last_step_name}' from Pipeline")
        logger.info(f"     Built-in scaler will be ignored - using our StandardScaler instead")

        return classifier

    def inspect_model(self, model):
        """Inspect model to understand what it expects."""
        logger.info("\n" + "=" * 80)
        logger.info("MODEL INSPECTION")
        logger.info("=" * 80)

        # Model type
        model_type = type(model).__name__
        logger.info(f"Model type: {model_type}")

        # Try to get feature information
        feature_info = {}

        # Check for sklearn Pipeline
        if hasattr(model, 'named_steps'):
            logger.info(f"Pipeline detected with steps: {list(model.named_steps.keys())}")
            feature_info['is_pipeline'] = True
            feature_info['pipeline_steps'] = list(model.named_steps.keys())

            # Check if pipeline has built-in scaler
            for step_name, step in model.named_steps.items():
                if 'scaler' in step_name.lower() or 'standard' in step_name.lower() or 'robust' in step_name.lower():
                    scaler_type = type(step).__name__
                    logger.info(f"  ⚠️  Pipeline has built-in scaler: {scaler_type}")
                    logger.info(f"     Our pipeline will use StandardScaler instead")
                    feature_info['has_builtin_scaler'] = True
                    feature_info['builtin_scaler_type'] = scaler_type

        # Check for sklearn models
        if hasattr(model, 'n_features_in_'):
            feature_info['n_features'] = model.n_features_in_
            logger.info(f"Expected features: {model.n_features_in_}")

        if hasattr(model, 'feature_names_in_'):
            feature_info['feature_names'] = model.feature_names_in_.tolist()
            logger.info(f"Feature names: {model.feature_names_in_}")

        # Check for tree-based models
        if hasattr(model, 'n_estimators'):
            logger.info(f"Number of estimators: {model.n_estimators}")

        # Check for feature importances
        if hasattr(model, 'feature_importances_'):
            feature_info['feature_importances'] = model.feature_importances_.tolist()
            logger.info(f"Has feature importances: Yes ({len(model.feature_importances_)} features)")

        logger.info("=" * 80)

        return model_type, feature_info

    def verify_compatibility(self, model, feature_info, test_csv: str):
        """Verify model is compatible with our preprocessing."""
        logger.info("\n" + "=" * 80)
        logger.info("COMPATIBILITY CHECK")
        logger.info("=" * 80)

        # Fit the pipeline on the dataset to learn transformation caps
        logger.info(f"Fitting preprocessing pipeline on {test_csv}...")
        df = pl.read_csv(test_csv)

        # Detect how many features the model expects
        n_features = None
        if 'n_features' in feature_info:
            n_features = feature_info['n_features']
            logger.info(f"Model expects: {n_features} features")

            # Determine preprocessing mode
            if n_features == 32:
                logger.info(f"  ℹ️  Using 32-feature mode (no sparse feature removal)")
            elif n_features == 28:
                logger.info(f"  ℹ️  Using 28-feature mode (with sparse feature removal)")
            else:
                logger.warning(f"  ⚠️  Unusual feature count: {n_features} (expected 28 or 32)")

        # Create pipeline with appropriate feature count
        self.pipeline = PDFPreprocessingPipeline(n_features=n_features)
        self.pipeline.fit(df)
        logger.info("  ✅ Pipeline fitted and caps learned")

        our_features = self.pipeline.get_feature_names()
        logger.info(f"Our preprocessing outputs: {len(our_features)} features")

        if n_features:
            if n_features == len(our_features):
                logger.info("✅ Feature count matches!")
            else:
                logger.warning(f"⚠️  Feature count mismatch! Model expects {n_features}, we provide {len(our_features)}")

        if 'feature_names' in feature_info:
            model_features = feature_info['feature_names']
            logger.info("\nFeature name comparison:")

            matches = []
            mismatches = []

            for i, (ours, theirs) in enumerate(zip(our_features, model_features)):
                if ours == theirs:
                    matches.append(ours)
                else:
                    mismatches.append(f"  [{i}] Ours: '{ours}' | Theirs: '{theirs}'")

            logger.info(f"✅ Matching features: {len(matches)}/{len(our_features)}")

            if mismatches:
                logger.warning(f"⚠️  Mismatched features: {len(mismatches)}")
                for mm in mismatches[:5]:  # Show first 5
                    logger.warning(mm)
                if len(mismatches) > 5:
                    logger.warning(f"  ... and {len(mismatches) - 5} more")

        logger.info("=" * 80)

    def test_model(self, model, test_csv: str = "datasets/PDF_All_features.csv"):
        """Test model on sample data."""
        logger.info("\n" + "=" * 80)
        logger.info("MODEL TESTING")
        logger.info("=" * 80)

        # Load test data
        logger.info(f"Loading test data from {test_csv}...")
        df = pl.read_csv(test_csv)
        logger.info(f"  Loaded {len(df)} samples")

        # Preprocess
        logger.info("Applying our preprocessing...")
        df_processed = self.pipeline.transform_dataframe(df)

        # Prepare test sample
        X_test = df_processed.drop("label").fill_null(0).to_numpy()[:10]  # First 10 samples
        y_test = df_processed["label"].to_numpy()[:10]

        # Predict
        logger.info("Making predictions...")
        try:
            predictions = model.predict(X_test)
            logger.info("  ✅ Predictions successful!")

            # Show results
            logger.info("\nSample predictions:")
            for i, (true, pred) in enumerate(zip(y_test, predictions)):
                status = "✅" if true == pred else "❌"
                true_label = "Malicious" if true == 1 else "Benign"
                pred_label = "Malicious" if pred == 1 else "Benign"
                logger.info(f"  {status} Sample {i}: True={true_label}, Predicted={pred_label}")

            # Accuracy on sample
            accuracy = (predictions == y_test).sum() / len(y_test)
            logger.info(f"\nSample accuracy: {accuracy:.1%}")

        except Exception as e:
            logger.error(f"  ❌ Prediction failed: {e}")
            raise

        logger.info("=" * 80)

    def save_model(self, model, model_type: str, feature_info: Dict):
        """Save model in our format."""
        logger.info(f"\nSaving model to {self.model_dir}...")

        # Create directory
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Save model (use joblib for consistency)
        model_path = self.model_dir / "model.joblib"
        joblib.dump(model, model_path)
        logger.info(f"  ✅ Model saved: {model_path}")

        # Save feature names
        features_path = self.model_dir / "features.json"
        with open(features_path, "w") as f:
            json.dump(self.pipeline.get_feature_names(), f, indent=2)
        logger.info(f"  ✅ Features saved: {features_path}")

        # Save transformation caps (critical for predictions!)
        caps_path = self.model_dir / "caps.json"
        with open(caps_path, "w") as f:
            json.dump(self.pipeline.get_caps(), f, indent=2)
        logger.info(f"  ✅ Transformation caps saved: {caps_path}")

        # Save scaler (critical for predictions!)
        scaler_path = self.model_dir / "scaler.joblib"
        joblib.dump(self.pipeline.get_scaler(), scaler_path)
        logger.info(f"  ✅ Scaler saved: {scaler_path}")

        # Save metadata
        metadata = {
            "import_date": datetime.now().isoformat(),
            "source": "pickle_import",
            "model_type": model_type,
            "features_count": len(self.pipeline.get_feature_names()),
            "feature_info": feature_info,
            "preprocessing": {
                "method": "Model3 - Winsorization + log1p + StandardScaler",
                "transformations": [
                    "Winsorization at 99th percentile (cap outliers)",
                    "log1p transformation (fix skewness)",
                    "StandardScaler (mean=0, std=1)"
                ],
                "sparse_removed": list(self.pipeline.selector.SPARSE_REMOVED) if not self.pipeline.skip_feature_selection else [],
                "caps_count": len(self.pipeline.get_caps()),
                "scaler_fitted": True,
                "n_features": self.pipeline.n_features,
                "mode": "32-feature (no sparse removal)" if self.pipeline.skip_feature_selection else "28-feature (sparse removal)"
            },
            "note": "Imported from pickle file. Uses Model3.ipynb preprocessing approach."
        }

        # Add notes about built-in preprocessing if detected
        if feature_info.get('has_builtin_scaler'):
            metadata['preprocessing']['note'] = (
                f"Original model had built-in {feature_info['builtin_scaler_type']}, "
                f"but we use our own StandardScaler for consistency."
            )

        metadata_path = self.model_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"  ✅ Metadata saved: {metadata_path}")

    def import_model(
        self,
        pickle_path: str,
        test: bool = True,
        test_csv: str = "datasets/PDF_All_features.csv"
    ):
        """Complete import workflow."""
        logger.info("=" * 80)
        logger.info("MODEL IMPORT")
        logger.info("=" * 80)

        # 1. Load model file (pickle or joblib)
        model = self.load_model_file(pickle_path)

        # 1b. Extract model from dict if necessary
        model = self.extract_model_from_dict(model)

        # 1c. Extract classifier from Pipeline if it has built-in scaler
        model = self.extract_classifier_from_pipeline(model)

        # 2. Inspect
        model_type, feature_info = self.inspect_model(model)

        # 3. Verify compatibility (fits pipeline on dataset)
        self.verify_compatibility(model, feature_info, test_csv)

        # 4. Test (optional)
        if test:
            self.test_model(model, test_csv)

        # 5. Save
        self.save_model(model, model_type, feature_info)

        logger.info("\n" + "=" * 80)
        logger.info("IMPORT COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"Model directory: {self.model_dir}")
        logger.info("\nYou can now use this model with:")
        logger.info(f"  python src/predict.py document.pdf")
        logger.info("=" * 80)


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Import trained model into pipeline",
        epilog="Example: python src/import_model.py model.joblib --test"
    )
    parser.add_argument(
        "model_path",
        help="Path to model file (.pkl, .pickle, or .joblib)"
    )
    parser.add_argument(
        "--test",
        "-t",
        action="store_true",
        help="Test model on sample data"
    )
    parser.add_argument(
        "--test-csv",
        default="datasets/PDF_All_features.csv",
        help="CSV file to use for testing"
    )

    args = parser.parse_args()

    # Run import
    try:
        importer = ModelImporter()
        importer.import_model(
            pickle_path=args.model_path,
            test=args.test,
            test_csv=args.test_csv
        )
        sys.exit(0)

    except Exception as e:
        logger.error(f"Import failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
