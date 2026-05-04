import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

def train_baseline(data_dir='data', models_dir='models'):
    print("Loading data...")
    train_path = os.path.join(data_dir, 'train.csv')
    val_path = os.path.join(data_dir, 'val.csv')
    
    if not os.path.exists(train_path):
        # Try parent directory for notebook support
        data_dir = os.path.join('..', data_dir)
        models_dir = os.path.join('..', models_dir)
        train_path = os.path.join(data_dir, 'train.csv')
        val_path = os.path.join(data_dir, 'val.csv')

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    # Fill NaN values if any
    train_df['text'] = train_df['text'].fillna('')
    val_df['text'] = val_df['text'].fillna('')
    
    X_train, y_train = train_df['text'], train_df['label']
    X_val, y_val = val_df['text'], val_df['label']
    
    print("Vectorizing text (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    
    print("Training Logistic Regression...")
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train_tfidf, y_train)
    
    print("Evaluating baseline model...")
    y_pred = model.predict(X_val_tfidf)
    
    print("\nAccuracy:", accuracy_score(y_val, y_pred))
    print("\nClassification Report:\n", classification_report(y_val, y_pred))
    
    # Save model and vectorizer
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    joblib.dump(model, os.path.join(models_dir, 'baseline_model.pkl'))
    joblib.dump(vectorizer, os.path.join(models_dir, 'tfidf_vectorizer.pkl'))
    print(f"Baseline model and vectorizer saved to {models_dir}/")

if __name__ == "__main__":
    if os.path.exists('data/train.csv'):
        train_baseline()
    else:
        print("Error: train.csv not found. Please run preprocessing.py first.")
