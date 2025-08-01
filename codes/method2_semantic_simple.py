import pickle
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import re
from collections import defaultdict
import warnings
import os

# Suppress warnings
warnings.filterwarnings("ignore", message=".*encoder_attention_mask.*")
warnings.filterwarnings("ignore", category=FutureWarning)

class SimplifiedSemanticSearch:
    def __init__(self, preprocessed_data_path='preprocessed_data.pkl', model_name='all-MiniLM-L6-v2'):
        """Initialize semantic search with sentence embeddings"""
        print("Loading preprocessed data...")
        with open(preprocessed_data_path, 'rb') as f:
            data = pickle.load(f)
        
        self.df = data['df']
        self.inverted_index = data['inverted_index']
        
        # Create a mapping from review_id to index for faster lookup
        self.review_id_to_idx = {rid: idx for idx, rid in enumerate(self.df['review_id'])}
        
        # Load sentence transformer model
        print(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Opinion lexicons
        self.positive_words = self.load_positive_lexicon()
        self.negative_words = self.load_negative_lexicon()
        
        print("Simplified semantic search engine initialized!")
    
    def load_opinion_lexicon(self, filepath):
        """Load opinion words from file"""
        words = set()
        
        if not os.path.exists(filepath):
            return words
        
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith(';'):
                    words.add(line.lower())
        
        return words
    
    def load_positive_lexicon(self):
        """Load positive opinion words"""
        file_words = self.load_opinion_lexicon('positive-words.txt')
        
        basic_words = set([
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'perfect', 'love', 'best', 'awesome', 'nice', 'super', 'brilliant',
            'outstanding', 'superior', 'exceptional', 'magnificent', 'terrific',
            'useful', 'helpful', 'strong', 'sharp', 'clear', 'fast', 'reliable',
            'solid', 'sturdy', 'efficient', 'effective', 'convenient', 'comfortable'
        ])
        
        return file_words.union(basic_words) if file_words else basic_words
    
    def load_negative_lexicon(self):
        """Load negative opinion words"""
        file_words = self.load_opinion_lexicon('negative-words.txt')
        
        basic_words = set([
            'bad', 'poor', 'terrible', 'awful', 'horrible', 'worst', 'hate',
            'disappointing', 'useless', 'broken', 'defective', 'weak', 'slow',
            'unreliable', 'uncomfortable', 'inconvenient', 'difficult', 'hard',
            'problem', 'issue', 'fail', 'failed', 'failure', 'error', 'wrong'
        ])
        
        return file_words.union(basic_words) if file_words else basic_words
    
    def parse_query(self, query):
        """Parse query in format 'aspect1 aspect2:opinion1 [opinion2]'"""
        parts = query.lower().split(':')
        if len(parts) != 2:
            raise ValueError("Query must be in format 'aspect:opinion'")
        
        aspect_terms = parts[0].strip().split()
        opinion_terms = parts[1].strip().split()
        
        return aspect_terms, opinion_terms
    
    def get_opinion_polarity(self, opinion_terms):
        """Determine if opinion terms are positive or negative"""
        pos_count = sum(1 for term in opinion_terms if term in self.positive_words)
        neg_count = sum(1 for term in opinion_terms if term in self.negative_words)
        
        if pos_count > neg_count:
            return 'positive'
        elif neg_count > pos_count:
            return 'negative'
        else:
            return 'neutral'
    
    def compute_text_similarity(self, text1, text2):
        """Compute similarity between two texts"""
        try:
            emb1 = self.model.encode([text1])[0]
            emb2 = self.model.encode([text2])[0]
            similarity = cosine_similarity([emb1], [emb2])[0][0]
            return similarity
        except:
            return 0.0
    
    def search_semantic_simple(self, query, top_k=100):
        """Simplified semantic search"""
        aspect_terms, opinion_terms = self.parse_query(query)
        
        # Get candidate reviews containing aspect terms
        aspect_reviews = set()
        for term in aspect_terms:
            if term in self.inverted_index:
                aspect_reviews.update(self.inverted_index[term])
        
        if not aspect_reviews:
            return []
        
        # Determine expected polarity
        expected_polarity = self.get_opinion_polarity(opinion_terms)
        
        # Score each candidate review
        scored_reviews = []
        
        for review_id in list(aspect_reviews)[:1000]:  # Limit to prevent memory issues
            # Get review by ID
            if review_id not in self.review_id_to_idx:
                continue
                
            idx = self.review_id_to_idx[review_id]
            review = self.df.iloc[idx]
            
            # Get review text
            review_text = str(review.get('review_text', '')).lower()
            if not review_text:
                continue
            
            # Basic scoring based on term presence
            score = 0
            
            # Check if aspect terms are present
            aspect_score = sum(1 for term in aspect_terms if term in review_text.split())
            
            # Check if opinion terms are present
            opinion_score = sum(1 for term in opinion_terms if term in review_text.split())
            
            # Rating consistency check
            rating = review.get('customer_review_rating', 3)
            rating_score = 0
            
            if expected_polarity == 'positive' and rating > 3:
                rating_score = 1
            elif expected_polarity == 'negative' and rating <= 3:
                rating_score = 1
            elif expected_polarity == 'neutral':
                rating_score = 0.5
            
            # Combine scores
            total_score = aspect_score + opinion_score + rating_score
            
            if total_score > 0:
                scored_reviews.append((review_id, total_score))
        
        # Sort by score and return top results
        scored_reviews.sort(key=lambda x: x[1], reverse=True)
        
        return [review_id for review_id, score in scored_reviews[:top_k]]
    
    def search_test4(self, query):
        """Test 4: Proper connotation using simplified semantic search"""
        return self.search_semantic_simple(query, top_k=100)
    
    def save_results(self, results, output_file):
        """Save results to text file"""
        with open(output_file, 'w') as f:
            for review_id in results:
                f.write(f"{review_id}\n")
        print(f"Saved {len(results)} results to {output_file}")

# Usage
if __name__ == "__main__":
    searcher = SimplifiedSemanticSearch()
    
    test_queries = [
        "audio quality:poor",
        "wifi signal:strong", 
        "mouse button:click problem",
        "gps map:useful",
        "image quality:sharp"
    ]
    
    import os
    os.makedirs("outputs/advanced_models", exist_ok=True)
    
    for query in test_queries:
        query_name = query.replace(" ", "_").replace(":", "_")
        
        # Run simplified search
        results = searcher.search_test4(query)
        searcher.save_results(results, f"outputs/advanced_models/{query_name}_test4.txt")
        
        print(f"Query: {query} - Found {len(results)} results")