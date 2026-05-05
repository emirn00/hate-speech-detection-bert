# Hate Speech Detection with Fine-Tuned BERT

This repository contains a full-stack application designed to detect and classify social media text into three categories: Hate Speech, Offensive Language, and Neutral Content. The core of the project is a Transformer-based (BERT) model fine-tuned on the Davidson Dataset, achieving high-precision classification through state-of-the-art NLP techniques.

## Key Features
- Transformer Architecture: Fine-tuned bert-base-uncased with custom classification heads.
- Robust Preprocessing: Custom pipeline for cleaning social media noise (mentions, URLs, punctuation).
- Imbalance Handling: Implemented class-weighting strategy to handle minority class detection in the dataset.
- Full-Stack Demo: FastAPI backend for real-time inference and an Angular-based UI for demonstration.
- Baseline Comparison: Includes a TF-IDF + Logistic Regression comparison for performance validation.

## Technology Stack
- Model Training: PyTorch, Transformers (HuggingFace), Scikit-Learn
- Backend: FastAPI, Uvicorn
- Frontend: Angular 17+
- Data Analysis: Pandas, Numpy, Matplotlib, Seaborn

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

## Setup and Installation

### 1. Clone the Repository
```bash
git clone https://github.com/emirn00/hate-speech-detection-bert
cd hate-speech-detection-bert
```

### 2. Python Environment (Backend and ML)
```bash
# Install dependencies
pip install -r requirements.txt

# Run model training (optional if weights exist)
python src/ml/bert_train.py
```

### 3. Frontend Setup
```bash
cd src/frontend
npm install
ng serve
```

## Evaluation and Sample Results

The fine-tuned BERT model outperforms traditional ML baselines in detecting the Hate Speech class, which is often the most challenging due to class imbalance.

### Sample Prediction
| Input Text | Prediction | Confidence |
| :--- | :--- | :--- |
| "I really enjoy the weather today." | Neutral | 99.1% |
| "You are such a loser, stay away from here." | Offensive | 94.5% |
| "I hate [Specific Group] and they should leave." | Hate Speech | 88.7% |

### Performance Metrics
- BERT F1-Score: ~0.91 (Weighted)
- Baseline F1-Score: ~0.71 (Macro)

## Running the Application
1. Start the backend:
   ```bash
   uvicorn src.backend.main:app --reload
   ```
2. Start the frontend:
   ```bash
   cd src/frontend
   ng serve
   ```
3. Access the application at http://localhost:4200.

## License and Credits
This project is developed for educational purposes as part of the LLM/NLP coursework. 
Data source: Davidson et al. (2017) Hate Speech and Offensive Language Dataset.
