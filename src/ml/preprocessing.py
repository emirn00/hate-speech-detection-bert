import pandas as pd
import re
import os
import requests
from sklearn.model_selection import train_test_split

def download_dataset(url, dest_path):
    """Downloads the dataset if it doesn't exist."""
    if not os.path.exists(dest_path):
        print(f"Downloading dataset from {url}...")
        response = requests.get(url)
        with open(dest_path, 'wb') as f:
            f.write(response.content)
        print("Download complete.")
    else:
        print("Dataset already exists.")

def clean_text(text):
    """
    Clean text:
    - lowercase
    - remove URLs
    - remove mentions (@user)
    - remove special characters (optional but usually good)
    """
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def prepare_data(file_path, output_dir='data'):
    """
    Load, clean, and split the dataset.
    Labels: 0 -> Hate Speech, 1 -> Offensive Language, 2 -> Neutral
    """
    df = pd.read_csv(file_path)
    
    # The Davidson dataset has 'tweet' and 'class' columns
    # We rename them to 'text' and 'label' for consistency
    if 'tweet' in df.columns:
        df = df.rename(columns={'tweet': 'text', 'class': 'label'})
    
    print("Cleaning text...")
    df['text'] = df['text'].apply(clean_text)
    
    # Select only necessary columns
    df = df[['text', 'label']]
    
    # Split: 70% Train, 15% Val, 15% Test
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['label'])
    
    # Save processed files
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    print(f"Data split complete: Train({len(train_df)}), Val({len(val_df)}), Test({len(test_df)})")
    return train_df, val_df, test_df

if __name__ == "__main__":
    DATA_URL = "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv"
    RAW_DATA_PATH = "data/labeled_data.csv"
    
    if not os.path.exists('data'):
        os.makedirs('data')
        
    download_dataset(DATA_URL, RAW_DATA_PATH)
    prepare_data(RAW_DATA_PATH)
