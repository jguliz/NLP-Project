import pickle
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
    
    def load_positive_lexicon(self):
        """Load positive opinion words - expand this with full lexicon"""
        # Basic positive words - you should load from the actual lexicon file
        return set([
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'perfect', 'love', 'best', 'awesome', 'nice', 'super', 'brilliant',
            'outstanding', 'superior', 'exceptional', 'magnificent', 'terrific',
            'useful', 'helpful', 'strong', 'sharp', 'clear', 'fast', 'reliable',
            'solid', 'sturdy', 'efficient', 'effective', 'convenient', 'comfortable',
            'positive_emoji'  # From preprocessing
        ])
    
    def load_negative_lexicon(self):
        """Load negative opinion words - expand this with full lexicon"""
        # Basic negative words - you should load from the actual lexicon file
        return set([
            'bad', 'poor', 'terrible', 'awful', 'horrible', 'worst', 'hate',
            'disappointing', 'useless', 'broken', 'defective', 'weak', 'slow',
            'unreliable', 'uncomfortable', 'inconvenient', 'difficult', 'hard',
            'problem', 'issue', 'fail', 'failed', 'failure', 'error', 'wrong',
            'negative_emoji'  # From preprocessing
        ])
    
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
        
        # Determine opinion polarity
        polarity = self.get_opinion_polarity(opinion_terms)
        
        # Get reviews containing aspect terms
        aspect_results = set()
        for term in aspect_terms:
            if term in self.inverted_index:
                aspect_results.update(self.inverted_index[term])
        
        # Get reviews containing opinion terms
        opinion_results = set()
        for term in opinion_terms:
            if term in self.inverted_index:
                opinion_results.update(self.inverted_index[term])
        
        # Intersection of aspect and opinion
        combined_results = aspect_results.intersection(opinion_results)
        
        # Filter by rating based on polarity
        filtered_results = []
        for review_id in combined_results:
            review = self.df[self.df['review_id'] == review_id].iloc[0]
            rating = review['customer_review_rating']
            
            if polarity == 'positive' and rating > 3:
                filtered_results.append(review_id)
            elif polarity == 'negative' and rating <= 3:
                filtered_results.append(review_id)
            elif polarity == 'neutral':
                # Include all ratings for neutral opinions
                filtered_results.append(review_id)
        
        return filtered_results
    
    def search_with_expanded_opinion(self, query):
        """Search with expanded opinion terms based on lexicon"""
        aspect_terms, opinion_terms = self.parse_query(query)
        
        # Expand opinion terms with synonyms from lexicon
        expanded_opinions = set(opinion_terms)
        
        # Add related positive/negative words based on polarity
        polarity = self.get_opinion_polarity(opinion_terms)
        if polarity == 'positive':
            # Add some related positive terms
            if 'good' in opinion_terms:
                expanded_opinions.update(['great', 'excellent', 'nice'])
            if 'strong' in opinion_terms:
                expanded_opinions.update(['solid', 'sturdy', 'reliable'])
            if 'useful' in opinion_terms:
                expanded_opinions.update(['helpful', 'convenient', 'effective'])
            if 'sharp' in opinion_terms:
                expanded_opinions.update(['clear', 'crisp', 'precise'])
        elif polarity == 'negative':
            # Add some related negative terms
            if 'poor' in opinion_terms:
                expanded_opinions.update(['bad', 'terrible', 'awful'])
            if 'problem' in opinion_terms:
                expanded_opinions.update(['issue', 'fail', 'error'])
        
        # Get reviews containing aspect terms
        aspect_results = set()
        for term in aspect_terms:
            if term in self.inverted_index:
                aspect_results.update(self.inverted_index[term])
        
        # Get reviews containing expanded opinion terms
        opinion_results = set()
        for term in expanded_opinions:
            if term in self.inverted_index:
                opinion_results.update(self.inverted_index[term])
        
        # Return intersection
        return list(aspect_results.intersection(opinion_results))
    
    def search_test4(self, query):
        """Test 4: Proper connotation - combine rating and lexicon approaches"""
        # Use rating-based filtering as primary method
        results = self.search_with_rating(query)
        
        # If too few results, try expanded opinion search
        if len(results) < 10:
            expanded_results = self.search_with_expanded_opinion(query)
            # Add new results that match the rating criteria
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
        
        # Run Method 1 search
        results = searcher.search_test4(query)
        searcher.save_results(results, f"outputs/method1/{query_name}_test4.txt")
        
        print(f"Query: {query} - Found {len(results)} results")