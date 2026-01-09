"""
PDF Phishing Detection

Usage:
    python src/predict.py document.pdf
    python src/predict.py document.pdf --json

Extracts features → preprocesses → predicts → done.
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Tuple
import joblib
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from features import PDFFeatureExtractor
from preprocessing import PDFPreprocessingPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PDFPhishingDetector:
    """Full pipeline - loads model and predicts on PDFs."""

    def __init__(self):
        self.model_dir = Path("models")

        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found at {self.model_dir}")

        self.extractor = PDFFeatureExtractor(use_ocr=False)  # OCR is slow
        self.pipeline = None  # Will be initialized after loading caps
        self.model = None
        self.metadata = None
        self.feature_names = None
        self.caps = None

        self._load_model()

    def _load_model(self):
        logger.info("Loading model...")

        model_path = self.model_dir / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model = joblib.load(model_path)

        features_path = self.model_dir / "features.json"
        if features_path.exists():
            with open(features_path, "r") as f:
                self.feature_names = json.load(f)

        # Load transformation caps (critical!)
        caps_path = self.model_dir / "caps.json"
        if not caps_path.exists():
            raise FileNotFoundError(f"Transformation caps not found: {caps_path}")
        with open(caps_path, "r") as f:
            self.caps = json.load(f)

        # Load scaler (critical!)
        scaler_path = self.model_dir / "scaler.joblib"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        scaler = joblib.load(scaler_path)

        # Load metadata to get n_features setting
        n_features = None
        metadata_path = self.model_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
                # Extract n_features from preprocessing config
                if 'preprocessing' in self.metadata and 'n_features' in self.metadata['preprocessing']:
                    n_features = self.metadata['preprocessing']['n_features']
                    logger.info(f"Using {self.metadata['preprocessing'].get('mode', 'default')} preprocessing mode")

        # Initialize pipeline with loaded caps, scaler, and n_features
        self.pipeline = PDFPreprocessingPipeline(caps=self.caps, scaler=scaler, n_features=n_features)

        logger.info("✅ Model loaded")

    def predict(self, pdf_path: str) -> Dict[str, any]:
        """Returns BENIGN or MALICIOUS with confidence score."""
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        logger.info(f"Analyzing: {pdf_path.name}")

        logger.info("  [1/3] Extracting features...")
        raw_features = self.extractor.extract(str(pdf_path))

        logger.info("  [2/3] Applying preprocessing...")
        X = self.pipeline.transform(raw_features)  # Returns scaled numpy array

        logger.info("  [3/3] Running model inference...")
        probabilities = self.model.predict_proba(X)[0]
        prediction_class = self.model.predict(X)[0]

        result = {
            'file': str(pdf_path),
            'prediction': 'MALICIOUS' if prediction_class == 1 else 'BENIGN',
            'confidence': float(max(probabilities)),
            'probabilities': {
                'benign': float(probabilities[0]),
                'malicious': float(probabilities[1])
            },
            'features_extracted': len(raw_features),
            'features_used': X.shape[1]  # Number of features after preprocessing
        }

        return result

    def predict_batch(self, pdf_paths: list) -> list:
        """
        Predict multiple PDFs.

        Args:
            pdf_paths: List of PDF file paths

        Returns:
            List of prediction results
        """
        results = []
        for pdf_path in pdf_paths:
            try:
                result = self.predict(pdf_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {pdf_path}: {e}")
                results.append({
                    'file': str(pdf_path),
                    'error': str(e)
                })
        return results


def format_result(result: Dict) -> str:
    """Format prediction result for display."""
    if 'error' in result:
        return f"❌ ERROR: {result['error']}"

    # Color codes
    if result['prediction'] == 'MALICIOUS':
        icon = "🚨"
        color = "\033[91m"  # Red
    else:
        icon = "✅"
        color = "\033[92m"  # Green

    reset = "\033[0m"

    output = []
    output.append("=" * 80)
    output.append(f"{icon} PREDICTION: {color}{result['prediction']}{reset}")
    output.append("=" * 80)
    output.append(f"File: {result['file']}")
    output.append(f"Confidence: {result['confidence']:.1%}")
    output.append(f"")
    output.append(f"Probabilities:")
    output.append(f"  Benign:    {result['probabilities']['benign']:.1%}")
    output.append(f"  Malicious: {result['probabilities']['malicious']:.1%}")
    output.append("=" * 80)

    return "\n".join(output)


def main():
    """Command-line interface for PDF phishing detection."""
    parser = argparse.ArgumentParser(
        description="Detect phishing/malicious PDFs using ML",
        epilog="Example: python src/predict.py suspicious_document.pdf"
    )
    parser.add_argument(
        "pdf_path",
        help="Path to PDF file to analyze"
    )
    parser.add_argument(
        "--json",
        "-j",
        action="store_true",
        help="Output result as JSON"
    )

    args = parser.parse_args()

    try:
        # Initialize detector
        detector = PDFPhishingDetector()

        # Make prediction
        result = detector.predict(args.pdf_path)

        # Output results
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(format_result(result))

        # Exit with appropriate code
        sys.exit(0 if result['prediction'] == 'BENIGN' else 1)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(2)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    main()
