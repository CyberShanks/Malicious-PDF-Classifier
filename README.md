# Malicious PDF Classification ML Pipeline

This repository is an aggregation of resources used to train a classifier and integrate within a ML pipeline.

### Install Requirements

```bash
# Create Virtual Environment
python3 -m venv venv
# Activate Virtual Environment
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### Navigate the Repository
#### Files
`DataCleaning.ipynb`: Contains code to clean the data

`ocr.ipynb`: Contains investigation of `used_ocr` feature

#### Folders
`/Pipeline`: Contains Pipeline Scripts and Trained Models.
- `/Pipeline/src`: Contains pipeline scripts
- `/Pipeline/models`: Contains trained models

To use the code, consult the `TESTING_GUIDE.md`, and fetch the dependencies from `/Pipeline/requirements.txt`

`/Resources`:
- Contains ydata.profiling output (dataset_profile.html) 
- Contains extractor code used by Dataset Author (extractor.ipynb)
- Presentation based on the Project (Presentation.pdf)
- Rest of the files are outputs from `/Scripts` contain PDF templates or harmless payloads.

`/Scripts`: Contains scripts to generate Malicious PDFs. Taken from [https://github.com/DidierStevens/DidierStevensSuite](https://github.com/DidierStevens/DidierStevensSuite)

`/dataset`: Contains the PDF dataset, citation of the dataset available below:

Nejati, N. et al. "A Comprehensive Multi-Format Malicious Attachment Dataset for Email Threat Detection."
Canadian Institute for Cybersecurity (CIC), University of New Brunswick, 2025.


