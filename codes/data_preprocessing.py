import pandas as pd
import re
import numpy as np
from collections import defaultdict
import pickle
import os

class DataPreprocessor:
    def __init__(self, data_path='reviews_segment.pkl'):
        """Initialize the preprocessor with data path"""
        self.data_path = data_path
        self.df = None
        self.stop_words = self.load_stop_words()
        
    def load_stop_words(self):
        """Load common stop words - you can expand this list"""
        # Basic stop words list - you should download the full list from the GitHub link
        return set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                    'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
                    'before', 'after', 'above', 'below', 'between', 'under', 'again',
                    'further', 'then', 'once', 'is', 'am', 'are', 'was', 'were', 'been',
                    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing'])
    
    def load_data(self):
        """Load the review data from pickle file"""
        print("Loading data...")
        self.df = pd.read_pickle(self.data_path)
        print(f"Loaded {len(self.df)} reviews")
        
        # Assuming the columns based on the SQL structure mentioned
        # You may need to adjust column names based on actual data
        print("Columns:", self.df.columns.tolist())
        return self.df
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Handle emoticons
        text = re.sub(r':\)', ' positive_emoji ', text)
        text = re.sub(r':-\)', ' positive_emoji ', text)
        text = re.sub(r':\(', ' negative_emoji ', text)
        text = re.sub(r':-\(', ' negative_emoji ', text)
        
        # Keep only alphanumeric and spaces
        text = re.sub(r'[^a-zA-Z0-9\s_]', ' ', text)
        
        # Remove extra spaces
        text = ' '.join(text.split())
        
        return text
    
    def tokenize(self, text):
        """Tokenize text into words"""
        return text.split()
    
    def preprocess_reviews(self):
        """Preprocess all reviews"""
        print("Preprocessing reviews...")
        
        # Ensure we have the rating column as numeric
        if 'customer_review_rating' in self.df.columns:
            self.df['customer_review_rating'] = pd.to_numeric(self.df['customer_review_rating'], errors='coerce')
            print(f"Rating distribution: {self.df['customer_review_rating'].value_counts().sort_index().to_dict()}")
        
        # Clean review text
        self.df['cleaned_text'] = self.df['review_text'].apply(self.clean_text)
        
        # Tokenize
        self.df['tokens'] = self.df['cleaned_text'].apply(self.tokenize)
        
        # Filter rare words (appearing less than 5 times)
        word_freq = defaultdict(int)
        for tokens in self.df['tokens']:
            for token in tokens:
                word_freq[token] += 1
        
        # Keep only words that appear at least 5 times
        self.valid_words = {word for word, freq in word_freq.items() if freq >= 5}
        
        # Filter tokens
        self.df['filtered_tokens'] = self.df['tokens'].apply(
            lambda tokens: [t for t in tokens if t in self.valid_words and t not in self.stop_words]
        )
        
        print("Preprocessing complete!")
        return self.df
    
    def create_inverted_index(self):
        """Create inverted index for efficient search"""
        print("Creating inverted index...")
        self.inverted_index = defaultdict(set)
        
        for idx, row in self.df.iterrows():
            review_id = row['review_id']
            for token in row['filtered_tokens']:
                self.inverted_index[token].add(review_id)
        
        print(f"Inverted index created with {len(self.inverted_index)} unique terms")
        return self.inverted_index
    
    def save_preprocessed_data(self, output_path='preprocessed_data.pkl'):
        """Save preprocessed data and indices"""
        data_to_save = {
            'df': self.df,
            'inverted_index': dict(self.inverted_index),
            'valid_words': self.valid_words
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(data_to_save, f)
        
        print(f"Preprocessed data saved to {output_path}")

# Usage
if __name__ == "__main__":
    preprocessor = DataPreprocessor('reviews_segment.pkl')
    df = preprocessor.load_data()
    df = preprocessor.preprocess_reviews()
    inverted_index = preprocessor.create_inverted_index()
    preprocessor.save_preprocessed_data()