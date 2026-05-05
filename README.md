# Hate Speech Detection with Fine-Tuned BERT

This repository contains a full-stack application designed to detect and classify social media text into three categories: Hate Speech, Offensive Language, and Neutral Content. The core of the project is a Transformer-based (BERT) model fine-tuned on the Davidson Dataset, achieving high-precision classification through state-of-the-art NLP techniques.

## Key Features
- Transformer Architecture: Fine-tuned bert-base-uncased with custom classification heads.
- Robust Preprocessing: Custom pipeline for cleaning social media noise (mentions, URLs, punctuation).
- Imbalance Handling: Implemented class-weighting strategy to handle minority class detection.
- Full-Stack Demo: FastAPI backend for real-time inference and an Angular-based UI for demonstration.

## Project Structure
```text
├── data/               # Labeled datasets (Davidson)
├── models/             # Saved model weights & tokenizers
├── notebooks/          # Exploratory Data Analysis & Training pipeline
├── src/
│   ├── ml/             # Core ML logic (BERT, Preprocessing, Baseline)
│   ├── backend/        # FastAPI Application
│   └── frontend/       # Angular UI
└── requirements.txt    # Python dependencies
```

## Required Libraries and Dependencies
The project requires Python 3.8+ and the following core libraries:
- **Transformers (HuggingFace):** Used for loading the pre-trained BERT model and tokenizer.
- **PyTorch:** The underlying deep learning framework for model fine-tuning and tensor operations.
- **Scikit-Learn:** Used for the TF-IDF baseline model, data splitting, and evaluation metrics.
- **FastAPI & Uvicorn:** Powering the backend REST API for model inference.
- **Angular:** Used for building the interactive web-based demonstration interface.

All Python dependencies can be installed via:
```bash
pip install -r requirements.txt
```

## Setup and Running Instructions

### 1. Data Preparation and Training
To replicate the results or retrain the model, follow this execution order:
1. **Preprocessing:** Run the script to download (if needed), clean, and split the raw data.
   ```bash
   python src/ml/preprocessing.py
   ```
2. **Training:** Fine-tune the BERT model. Note: This requires a GPU for reasonable training times.
   ```bash
   python src/ml/bert_train.py
   ```

### 2. Running the Full-Stack Application
1. **Backend:** Start the FastAPI server.
   ```bash
   # From the project root
   uvicorn src.backend.main:app --reload
   ```
2. **Frontend:** Start the Angular development server.
   ```bash
   cd src/frontend
   npm install
   ng serve
   ```
3. Access the application at `http://localhost:4200`.

## Sample Inputs and Outputs
The system accepts raw text strings and returns the predicted category along with a confidence score.

| Sample Input | Expected Output | Rationale |
| :--- | :--- | :--- |
| "I really enjoy the weather today." | **Neutral** | No hateful or offensive tokens detected. |
| "This is so stupid, I can't believe it." | **Offensive** | Contains derogatory terms but lacks targeted hate intent. |
| "I hate [Specific Group] and they should leave." | **Hate Speech** | Contains targeted attack and dehumanizing language. |

## Evaluation Results
- **BERT F1-Score:** ~0.91 (Weighted)
- **Baseline (LogReg) F1-Score:** ~0.71 (Macro)

## Credits
Data source: Davidson et al. (2017) Hate Speech and Offensive Language Dataset. Developed for the LLM Final Project submission.
