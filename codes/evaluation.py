import pandas as pd
import random
import os

class PrecisionEvaluator:
    def __init__(self, preprocessed_data_path='preprocessed_data.pkl'):
        """Initialize evaluator with review data"""
        import pickle
        with open(preprocessed_data_path, 'rb') as f:
            data = pickle.load(f)
        self.df = data['df']
        
    def evaluate_results(self, results_file, query, sample_size=50):
        """Manually evaluate relevance of retrieved results"""
        # Load retrieved review IDs
        with open(results_file, 'r') as f:
            review_ids = [line.strip() for line in f.readlines()]
        
        print(f"\nEvaluating query: {query}")
        print(f"Total retrieved: {len(review_ids)}")
        
        # Sample if too many results
        if len(review_ids) > sample_size:
            print(f"Sampling {sample_size} reviews for evaluation...")
            sampled_ids = random.sample(review_ids, sample_size)
        else:
            sampled_ids = review_ids
        
        relevant_count = 0
        aspect, opinion = query.split(':')
        
        print(f"\nFor each review, determine if it mentions '{aspect}' with '{opinion}' sentiment")
        print("Enter 'y' for relevant, 'n' for not relevant, 'q' to quit\n")
        
        for i, review_id in enumerate(sampled_ids):
            review = self.df[self.df['review_id'] == review_id].iloc[0]
            print(f"\n--- Review {i+1}/{len(sampled_ids)} (ID: {review_id}) ---")
            print(f"Rating: {review['customer_review_rating']} stars")
            print(f"Text: {review['review_text'][:500]}...")  # First 500 chars
            
            while True:
                response = input("\nIs this relevant? (y/n/q): ").lower()
                if response == 'y':
                    relevant_count += 1
                    break
                elif response == 'n':
                    break
                elif response == 'q':
                    if i > 0:
                        precision = relevant_count / (i + 1)
                        print(f"\nPartial precision: {precision:.3f} ({relevant_count}/{i+1})")
                    return
            
        precision = relevant_count / len(sampled_ids)
        
        if len(review_ids) > sample_size:
            print(f"\nSampled precision: {precision:.3f} ({relevant_count}/{sample_size})")
            estimated_relevant = int(precision * len(review_ids))
            print(f"Estimated total relevant: {estimated_relevant}/{len(review_ids)}")
        else:
            print(f"\nPrecision: {precision:.3f} ({relevant_count}/{len(review_ids)})")
            
        return precision, relevant_count, len(sampled_ids)

if __name__ == "__main__":
    evaluator = PrecisionEvaluator()
    
    queries = [
        "audio quality:poor",
        "wifi signal:strong",
        "mouse button:click problem",
        "gps map:useful",
        "image quality:sharp"
    ]
    
    for query in queries:
        query_name = query.replace(" ", "_").replace(":", "_")
        
        print("\n" + "="*50)
        print("BASELINE EVALUATION")
        evaluator.evaluate_results(
            f"outputs/baseline/{query_name}_test2.txt",
            query,
            sample_size=30
        )