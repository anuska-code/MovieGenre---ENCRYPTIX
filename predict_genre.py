import pickle
import joblib
import numpy as np
from movie_genre_classifier import preprocess_text
import nltk

def save_model_components(model, tfidf, mlb):
    """Save the trained model and its components"""
    joblib.dump(model, 'model.joblib')
    joblib.dump(tfidf, 'tfidf.joblib')
    joblib.dump(mlb, 'mlb.joblib')

def load_model_components():
    """Load the trained model and its components"""
    model = joblib.load('model.joblib')
    tfidf = joblib.load('tfidf.joblib')
    mlb = joblib.load('mlb.joblib')
    return model, tfidf, mlb

def predict_genre(plot_summary, threshold=0.3):
    """Predict genres for a given plot summary"""
    try:
        # Load the model components
        model, tfidf, mlb = load_model_components()
        
        # Preprocess the input plot summary
        processed_summary = preprocess_text(plot_summary)
        
        # Transform text using the trained TF-IDF vectorizer
        X = tfidf.transform([processed_summary])
        
        # Get probability predictions
        pred_probs = model.predict_proba(X)
        
        # Convert probabilities to predictions using threshold
        predictions = (pred_probs >= threshold).astype(int)
        
        # If no genres predicted, take top 2 most likely genres
        if predictions[0].sum() == 0:
            top_genres = np.argsort(pred_probs[0])[-2:]  # Get indices of top 2 genres
            predictions[0, top_genres] = 1
        
        # Convert prediction to genre labels
        genres = mlb.inverse_transform(predictions)[0]
        
        # Get confidence scores for predicted genres
        genre_probs = {genre: float(pred_probs[0][i]) 
                      for i, genre in enumerate(mlb.classes_) 
                      if predictions[0][i] == 1}
        
        return list(genres), genre_probs
    
    except Exception as e:
        print(f"Error predicting genre: {e}")
        return ['Error occurred'], {}

if __name__ == "__main__":
    print("Enter a movie plot summary (press Enter twice to finish):")
    lines = []
    while True:
        line = input()
        if line:
            lines.append(line)
        else:
            break
    plot_summary = ' '.join(lines)
    
    if plot_summary.strip():
        print("\nPredicting genres...")
        genres, probabilities = predict_genre(plot_summary)
        print("\nPredicted genres:")
        for genre in genres:
            confidence = probabilities.get(genre, 0) * 100
            print(f"- {genre} (Confidence: {confidence:.1f}%)")
    else:
        print("No plot summary provided.")
