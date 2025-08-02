import pickle
import os
import pandas as pd
from collections import defaultdict

class RatingLexiconSearch:
    def __init__(self, preprocessed_data_path='preprocessed_data.pkl'):
        """Initialize search with rating and lexicon enhancement"""
        print("Loading preprocessed data...")
        with open(preprocessed_data_path, 'rb') as f:
            data = pickle.load(f)
        
        self.df = data['df']
        self.inverted_index = data['inverted_index']
        self.valid_words = data['valid_words']
        
        # Initialize opinion lexicon (you should download the full lexicon from Hu and Liu)
        self.positive_words = self.load_positive_lexicon()
        self.negative_words = self.load_negative_lexicon()
        
        print("Data and lexicons loaded successfully!")
    
    def load_opinion_lexicon(self, filepath):
        """Load opinion words from file"""
        words = set()
        
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found, using basic word list")
            return words
        
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith(';'):
                    words.add(line.lower())
        
        print(f"Loaded {len(words)} words from {filepath}")
        return words

    def load_positive_lexicon(self):
        """Load positive opinion words"""
        file_words = self.load_opinion_lexicon('positive-words.txt')
        return file_words
    
    def load_negative_lexicon(self):
        """Load negative opinion words"""
        file_words = self.load_opinion_lexicon('negative-words.txt')
        return file_words
    
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
    
    def parse_query(self, query):
        """Parse query in format 'aspect1 aspect2:opinion1 [opinion2]'"""
        parts = query.lower().split(':')
        if len(parts) != 2:
            raise ValueError("Query must be in format 'aspect:opinion'")
        
        aspect_terms = parts[0].strip().split()
        opinion_terms = parts[1].strip().split()
        
        return aspect_terms, opinion_terms
    
    def search_with_rating(self, query):
        """Search combining boolean search with rating filtering"""
        aspect_terms, opinion_terms = self.parse_query(query)
        
        polarity = self.get_opinion_polarity(opinion_terms)
        
        aspect_results = set()
        for term in aspect_terms:
            if term in self.inverted_index:
                aspect_results.update(self.inverted_index[term])
        
        opinion_results = set()
        for term in opinion_terms:
            if term in self.inverted_index:
                opinion_results.update(self.inverted_index[term])
        
        combined_results = aspect_results.intersection(opinion_results)
        
        filtered_results = []
        for review_id in combined_results:
            review = self.df[self.df['review_id'] == review_id].iloc[0]
            rating = review['customer_review_rating']
            
            if polarity == 'positive' and rating > 3:
                filtered_results.append(review_id)
            elif polarity == 'negative' and rating <= 3:
                filtered_results.append(review_id)
            elif polarity == 'neutral':
                filtered_results.append(review_id)
        
        return filtered_results
    
    def search_with_expanded_opinion(self, query):
        """Search with expanded opinion terms based on lexicon"""
        aspect_terms, opinion_terms = self.parse_query(query)
        
        expanded_opinions = set(opinion_terms)
        
        polarity = self.get_opinion_polarity(opinion_terms)
        if polarity == 'positive':
            if 'good' in opinion_terms:
                expanded_opinions.update(['great', 'excellent', 'nice'])
            if 'strong' in opinion_terms:
                expanded_opinions.update(['solid', 'sturdy', 'reliable'])
            if 'useful' in opinion_terms:
                expanded_opinions.update(['helpful', 'convenient', 'effective'])
            if 'sharp' in opinion_terms:
                expanded_opinions.update(['clear', 'crisp', 'precise'])
        elif polarity == 'negative':
            if 'poor' in opinion_terms:
                expanded_opinions.update(['bad', 'terrible', 'awful'])
            if 'problem' in opinion_terms:
                expanded_opinions.update(['issue', 'fail', 'error'])
        
        aspect_results = set()
        for term in aspect_terms:
            if term in self.inverted_index:
                aspect_results.update(self.inverted_index[term])
        
        opinion_results = set()
        for term in expanded_opinions:
            if term in self.inverted_index:
                opinion_results.update(self.inverted_index[term])
        
        return list(aspect_results.intersection(opinion_results))
    
    def search_test4(self, query):
        """Test 4: Proper connotation - combine rating and lexicon approaches"""
        results = self.search_with_rating(query)
        
        if len(results) < 10:
            expanded_results = self.search_with_expanded_opinion(query)
            aspect_terms, opinion_terms = self.parse_query(query)
            polarity = self.get_opinion_polarity(opinion_terms)
            
            for review_id in expanded_results:
                if review_id not in results:
                    review = self.df[self.df['review_id'] == review_id].iloc[0]
                    rating = review['customer_review_rating']
                    
                    if polarity == 'positive' and rating > 3:
                        results.append(review_id)
                    elif polarity == 'negative' and rating <= 3:
                        results.append(review_id)
        
        return results
    
    def save_results(self, results, output_file):
        """Save results to text file"""
        with open(output_file, 'w') as f:
            for review_id in results:
                f.write(f"{review_id}\n")
        print(f"Saved {len(results)} results to {output_file}")

# Usage
if __name__ == "__main__":
    searcher = RatingLexiconSearch()
    
    test_queries = [
        "audio quality:poor",
        "wifi signal:strong", 
        "mouse button:click problem",
        "gps map:useful",
        "image quality:sharp"
    ]
    
    import os
    os.makedirs("outputs/method1", exist_ok=True)
    
    for query in test_queries:
        query_name = query.replace(" ", "_").replace(":", "_")
        
        results = searcher.search_test4(query)
        searcher.save_results(results, f"outputs/method1/{query_name}_test4.txt")
        
        print(f"Query: {query} - Found {len(results)} results")