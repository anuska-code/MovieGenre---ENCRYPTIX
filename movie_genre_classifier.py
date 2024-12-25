import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import joblib

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    if pd.isna(text):
        return ""
    # Convert to lowercase
    text = str(text).lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into text
    return ' '.join(tokens)

def load_data(train_file, test_file):
    try:
        # Load training data
        train_data = pd.read_csv(train_file, sep=':::', names=['ID', 'TITLE', 'GENRE', 'DESCRIPTION'], engine='python')
        
        # Load test data
        test_data = pd.read_csv(test_file, sep=':::', names=['ID', 'TITLE', 'DESCRIPTION'], engine='python')
        
        return train_data, test_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def prepare_data(train_data, test_data):
    try:
        # Preprocess descriptions
        print("Preprocessing training descriptions...")
        train_data['processed_description'] = train_data['DESCRIPTION'].apply(preprocess_text)
        print("Preprocessing test descriptions...")
        test_data['processed_description'] = test_data['DESCRIPTION'].apply(preprocess_text)
        
        # Convert genres to multi-label format
        print("Processing genres...")
        mlb = MultiLabelBinarizer()
        # Split genres and strip whitespace
        train_data['GENRE'] = train_data['GENRE'].str.strip().str.split(',')
        train_data['GENRE'] = train_data['GENRE'].apply(lambda x: [g.strip() for g in x])
        y = mlb.fit_transform(train_data['GENRE'])
        
        # Create TF-IDF features
        print("Creating TF-IDF features...")
        tfidf = TfidfVectorizer(max_features=5000, min_df=2, ngram_range=(1, 2))
        X_train = tfidf.fit_transform(train_data['processed_description'])
        X_test = tfidf.transform(test_data['processed_description'])
        
        return X_train, y, X_test, mlb, tfidf
    except Exception as e:
        print(f"Error preparing data: {e}")
        return None, None, None, None, None

def train_model(X_train, y):
    try:
        # Using OneVsRestClassifier with LogisticRegression for multi-label classification
        print("Training model...")
        model = OneVsRestClassifier(LogisticRegression(max_iter=1000, C=1.0))
        model.fit(X_train, y)
        return model
    except Exception as e:
        print(f"Error training model: {e}")
        return None

def predict_genres(model, X, mlb, threshold=0.3):
    # Get probability predictions
    pred_probs = model.predict_proba(X)
    
    # Convert probabilities to predictions using threshold
    predictions = (pred_probs >= threshold).astype(int)
    
    # If no genres predicted, take top 2 most likely genres
    for i, pred in enumerate(predictions):
        if pred.sum() == 0:
            top_genres = np.argsort(pred_probs[i])[-2:]  # Get indices of top 2 genres
            predictions[i, top_genres] = 1
    
    return predictions

def main():
    try:
        # Load data
        print("Loading data...")
        train_data, test_data = load_data('train_data.txt', 'test_data.txt')
        
        if train_data is None or test_data is None:
            return
        
        # Prepare features and labels
        print("Preparing data...")
        X_train, y, X_test, mlb, tfidf = prepare_data(train_data, test_data)
        
        if X_train is None or y is None or X_test is None or mlb is None or tfidf is None:
            return
        
        # Train model
        model = train_model(X_train, y)
        
        if model is None:
            return
        
        # Save model components
        print("Saving model components...")
        joblib.dump(model, 'model.joblib')
        joblib.dump(tfidf, 'tfidf.joblib')
        joblib.dump(mlb, 'mlb.joblib')
        
        # Make predictions on test data
        print("Making predictions...")
        predictions = predict_genres(model, X_test, mlb)
        
        # Convert predictions to genre labels
        predicted_genres = mlb.inverse_transform(predictions)
        
        # Save predictions
        print("Saving predictions...")
        with open('predictions.txt', 'w', encoding='utf-8') as f:
            for idx, genres in zip(test_data['ID'], predicted_genres):
                genre_str = ','.join(genres)
                f.write(f"{idx}:::{genre_str}\n")
        
        print("Done! Predictions saved to predictions.txt")
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()
