import re
import pickle
from collections import defaultdict

class BaselineBooleanSearch:
    def __init__(self, preprocessed_data_path='preprocessed_data.pkl'):
        """Initialize the baseline search engine"""
        print("Loading preprocessed data...")
        with open(preprocessed_data_path, 'rb') as f:
            data = pickle.load(f)
        
        self.df = data['df']
        self.inverted_index = data['inverted_index']
        self.valid_words = data['valid_words']
        print("Data loaded successfully!")
    
    def parse_query(self, query):
        """Parse query in format 'aspect1 aspect2:opinion1 [opinion2]'"""
        parts = query.lower().split(':')
        if len(parts) != 2:
            raise ValueError("Query must be in format 'aspect:opinion'")
        
        aspect_terms = parts[0].strip().split()
        opinion_terms = parts[1].strip().split()
        
        return aspect_terms, opinion_terms
    
    def word_search(self, word):
        """Search for exact word matches (not substrings)"""
        # Use word boundaries to avoid substring matches
        results = set()
        
        # Check if word exists in inverted index
        if word in self.inverted_index:
            results.update(self.inverted_index[word])
        
        return results
    
    def boolean_search(self, query, search_type='or_and'):
        """
        Perform boolean search based on different strategies
        search_type options:
        - 'or_and': (aspect1 OR aspect2) AND (opinion1 OR opinion2)
        - 'and_or': (aspect1 AND opinion1) OR (aspect2 AND opinion1) OR ...
        - 'any': Any term matches
        """
        aspect_terms, opinion_terms = self.parse_query(query)
        
        if search_type == 'or_and':
            # (aspect1 OR aspect2) AND (opinion1 OR opinion2)
            aspect_results = set()
            for term in aspect_terms:
                aspect_results.update(self.word_search(term))
            
            opinion_results = set()
            for term in opinion_terms:
                opinion_results.update(self.word_search(term))
            
            # Return intersection (AND)
            if aspect_results and opinion_results:
                return aspect_results.intersection(opinion_results)
            else:
                return set()
        
        elif search_type == 'and_or':
            # (aspect1 AND opinion1) OR (aspect1 AND opinion2) OR ...
            all_results = set()
            
            for aspect in aspect_terms:
                aspect_set = self.word_search(aspect)
                for opinion in opinion_terms:
                    opinion_set = self.word_search(opinion)
                    if aspect_set and opinion_set:
                        all_results.update(aspect_set.intersection(opinion_set))
            
            return all_results
        
        elif search_type == 'any':
            # Any term matches
            all_results = set()
            all_terms = aspect_terms + opinion_terms
            
            for term in all_terms:
                all_results.update(self.word_search(term))
            
            return all_results
        
        else:
            raise ValueError(f"Unknown search type: {search_type}")
    
    def search_test1(self, query):
        """Test 1: Review must contain at least one aspect word"""
        aspect_terms, _ = self.parse_query(query)
        
        results = set()
        for term in aspect_terms:
            results.update(self.word_search(term))
        
        return list(results)
    
    def search_test2(self, query):
        """Test 2: Review must mention both aspect and opinion terms"""
        return list(self.boolean_search(query, search_type='or_and'))
    
    def search_test3(self, query):
        """Test 3: Review may mention either aspect or opinion"""
        return list(self.boolean_search(query, search_type='any'))
    
    def save_results(self, results, output_file):
        """Save results to text file"""
        with open(output_file, 'w') as f:
            for review_id in results:
                f.write(f"{review_id}\n")
        print(f"Saved {len(results)} results to {output_file}")
    
    def evaluate_precision(self, retrieved, relevant):
        """Calculate precision metric"""
        if len(retrieved) == 0:
            return 0.0
        
        relevant_retrieved = len(set(retrieved).intersection(set(relevant)))
        precision = relevant_retrieved / len(retrieved)
        return precision

# Usage
if __name__ == "__main__":
    # Initialize search engine
    searcher = BaselineBooleanSearch()
    
    # Test queries
    test_queries = [
        "audio quality:poor",
        "wifi signal:strong",
        "mouse button:click problem",
        "gps map:useful",
        "image quality:sharp"
    ]
    
    # Create output directories
    import os
    os.makedirs("outputs/baseline", exist_ok=True)
    
    # Run searches for all test types
    for query in test_queries:
        query_name = query.replace(" ", "_").replace(":", "_")
        
        # Test 1: Aspect only
        results_test1 = searcher.search_test1(query)
        searcher.save_results(results_test1, f"outputs/baseline/{query_name}_test1.txt")
        
        # Test 2: Aspect AND Opinion
        results_test2 = searcher.search_test2(query)
        searcher.save_results(results_test2, f"outputs/baseline/{query_name}_test2.txt")
        
        # Test 3: Aspect OR Opinion
        results_test3 = searcher.search_test3(query)
        searcher.save_results(results_test3, f"outputs/baseline/{query_name}_test3.txt")
        
        print(f"\nQuery: {query}")
        print(f"Test 1 (Aspect only): {len(results_test1)} results")
        print(f"Test 2 (AND): {len(results_test2)} results")
        print(f"Test 3 (OR): {len(results_test3)} results")