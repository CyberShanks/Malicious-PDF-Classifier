# Pipeline Testing Guide

## Step 1: Re-import Model Properly

Your current model is missing critical preprocessing files. Re-import it:

```bash
# Re-import the model (this will regenerate caps.json and scaler.joblib)
python src/import_model.py /path/to/model.joblib --test --test-csv datasets/PDF_All_features.csv
```

**What this does**:
1. Loads your existing model
2. Fits the preprocessing pipeline on the training dataset
3. Generates caps.json (Winsorization parameters)
4. Generates scaler.joblib (StandardScaler parameters)
5. Tests compatibility and predictions
6. Saves everything to models/

**Verify all files exist**:
```bash
ls -lh models/
```

You should see 5 files now.

---

## Step 2: Test Feature Extraction

Test that feature extraction works on a sample PDF.

### Get a test PDF

**Option 1**: Use a matplotlib PDF (simple, benign):
```bash
cp .venv/lib/python3.12/site-packages/matplotlib/mpl-data/images/matplotlib.pdf test_benign.pdf
```

**Option 2**: Create a simple test PDF:
```bash
python3 << 'EOF'
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

c = canvas.Canvas("test_benign.pdf", pagesize=letter)
c.drawString(100, 750, "This is a test PDF")
c.drawString(100, 730, "Used for testing the pipeline")
c.save()
print("✅ Created test_benign.pdf")
EOF
```

If reportlab is not installed:
```bash
pip install reportlab
```

### Test feature extraction directly:

```bash
python3 << 'EOF'
import sys
sys.path.insert(0, 'src')
from features import PDFFeatureExtractor
import json

# Extract features
extractor = PDFFeatureExtractor(use_ocr=False)
features = extractor.extract("test_benign.pdf")

# Display results
print(f"✅ Extracted {len(features)} features")
print("\nSample features:")
for key in list(features.keys())[:10]:
    print(f"  {key}: {features[key]}")

# Save for inspection
with open("extracted_features.json", "w") as f:
    json.dump(features, f, indent=2)
print("\n✅ Features saved to: extracted_features.json")
EOF
```

**Expected output**:
```
✅ Extracted 41 features

Sample features:
  file_path: test_benign.pdf
  file_size: 1234
  page_count: 1
  valid_pdf_header: 1
  ...

✅ Features saved to: extracted_features.json
```

**Troubleshooting**:
- If extraction fails, check PDF file exists: `ls -la test_benign.pdf`
- Check PyMuPDF is installed: `pip list | grep PyMuPDF`

---

## Step 3: Test Preprocessing Pipeline

Test that preprocessing transforms features correctly.

```bash
python3 << 'EOF'
import sys
import json
import joblib
sys.path.insert(0, 'src')
from preprocessing import PDFPreprocessingPipeline

# Load preprocessing artifacts
with open("models/caps.json", "r") as f:
    caps = json.load(f)
scaler = joblib.load("models/scaler.joblib")

# Load raw features
with open("extracted_features.json", "r") as f:
    raw_features = json.load(f)

# Initialize pipeline with saved parameters
pipeline = PDFPreprocessingPipeline(caps=caps, scaler=scaler)

# Transform
X = pipeline.transform(raw_features)

print(f"✅ Preprocessing successful!")
print(f"Input: {len(raw_features)} raw features (dict)")
print(f"Output: {X.shape} scaled features (numpy array)")
print(f"\nFirst 5 values: {X[0][:5]}")
EOF
```

**Expected output**:
```
✅ Preprocessing successful!
Input: 41 raw features (dict)
Output: (1, 28) scaled features (numpy array)

First 5 values: [-0.123 0.456 -0.789 0.234 -0.567]
```

**Troubleshooting**:
- If caps.json missing: Re-run Step 1
- If scaler.joblib missing: Re-run Step 1
- If values look wrong: Check that caps and scaler are from same training run

---

## Step 4: Test Full Prediction Pipeline

Test the complete end-to-end prediction.

```bash
python src/predict.py test_benign.pdf
```

**Expected output**:
```
INFO - Loading model...
INFO - ✅ Model loaded
INFO - Analyzing: test_benign.pdf
INFO -   [1/3] Extracting features...
INFO -   [2/3] Applying preprocessing...
INFO -   [3/3] Running model inference...
================================================================================
✅ PREDICTION: BENIGN
================================================================================
File: test_benign.pdf
Confidence: 99.8%

Probabilities:
  Benign:    99.8%
  Malicious: 0.2%
================================================================================
```

**Test JSON output**:
```bash
python src/predict.py test_benign.pdf --json
```

**Expected output**:
```json
{
  "file": "test_benign.pdf",
  "prediction": "BENIGN",
  "confidence": 0.998,
  "probabilities": {
    "benign": 0.998,
    "malicious": 0.002
  },
  "features_extracted": 41,
  "features_used": 28
}
```

**Troubleshooting**:
- If "Model file not found": Check `models/model.joblib` exists
- If "Scaler not found": Re-run Step 1
- If "Caps not found": Re-run Step 1
- If prediction fails: Check logs for detailed error

---

## Step 5: Test with Real PDFs

To properly test the model, you need both benign and malicious PDFs.

### Get Test PDFs

**Benign PDFs** (safe sources):
- Download from: https://www.adobe.com/support/products/enterprise/knowledgecenter/media/c4611_sample_explain.pdf
- Or use matplotlib PDFs: `cp .venv/lib/python3.12/site-packages/matplotlib/mpl-data/images/*.pdf ./`

**Malicious PDFs** (use caution):
- Check if your training dataset includes sample file paths
- Use samples from your original dataset (if you have the PDFs)
- Use public malware repositories (with proper security precautions)
---

## Step 6: Test Error Handling

Verify the pipeline handles errors gracefully.

### Test with non-existent file:
```bash
python src/predict.py nonexistent.pdf
```

**Expected**: Clear error message, not a crash

### Test with non-PDF file:
```bash
echo "not a pdf" > fake.pdf
python src/predict.py fake.pdf
```

**Expected**: Should handle gracefully (may extract minimal features)

### Test with corrupted model:
```bash
# Backup model
cp models/model.joblib models/model.joblib.bak

# Corrupt it
echo "corrupted" > models/model.joblib

# Try to load
python src/predict.py test_benign.pdf
```

**Expected**: Clear error message about model loading

**Restore**:
```bash
mv models/model.joblib.bak models/model.joblib
```

## Quick Test Checklist

Use this checklist to verify everything works:

- [ ] All 5 model files exist in `models/`
- [ ] Feature extraction works (`python -c "from src.features import PDFFeatureExtractor; e=PDFFeatureExtractor(); print(e.extract('test_benign.pdf'))"`)
- [ ] Preprocessing works (run Step 3)
- [ ] Prediction works (`python src/predict.py test_benign.pdf`)
- [ ] JSON output works (`python src/predict.py test_benign.pdf --json`)

## Quick Command Reference

```bash
# Re-import model
python src/import_model.py models/model.joblib --test

# Single prediction
python src/predict.py document.pdf

# JSON output
python src/predict.py document.pdf --json

```
