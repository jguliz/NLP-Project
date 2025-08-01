"""
Main pipeline to run all methods and generate evaluation results
"""

import os
import pandas as pd
from data_preprocessing import DataPreprocessor
from baseline_boolean import BaselineBooleanSearch
from method1_rating_lexicon import RatingLexiconSearch
from method2_semantic_simple import SimplifiedSemanticSearch

def create_output_directories():
    """Create necessary output directories"""
    directories = [
        "outputs",
        "outputs/baseline",
        "outputs/method1", 
        "outputs/advanced_models",
        "codes"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("Output directories created!")

def run_baseline_searches(searcher, queries):
    """Run baseline boolean searches for all test types"""
    results = {}
    
    for query in queries:
        query_name = query.replace(" ", "_").replace(":", "_")
        results[query] = {}
        
        # Test 1: Aspect only
        results_test1 = searcher.search_test1(query)
        searcher.save_results(results_test1, f"outputs/baseline/{query_name}_test1.txt")
        results[query]['test1'] = len(results_test1)
        
        # Test 2: Aspect AND Opinion
        results_test2 = searcher.search_test2(query)
        searcher.save_results(results_test2, f"outputs/baseline/{query_name}_test2.txt")
        results[query]['test2'] = len(results_test2)
        
        # Test 3: Aspect OR Opinion
        results_test3 = searcher.search_test3(query)
        searcher.save_results(results_test3, f"outputs/baseline/{query_name}_test3.txt")
        results[query]['test3'] = len(results_test3)
    
    return results

def evaluate_results(method_name, queries, searcher):
    """Evaluate results for a method"""
    print(f"\n{'='*50}")
    print(f"Evaluating {method_name}")
    print(f"{'='*50}")
    
    evaluation_data = []
    
    for query in queries:
        query_name = query.replace(" ", "_").replace(":", "_")
        
        # For demonstration, we'll calculate metrics
        # In practice, you'd need ground truth labels
        if method_name == "Baseline":
            # Run all three tests
            test1_results = searcher.search_test1(query)
            test2_results = searcher.search_test2(query) 
            test3_results = searcher.search_test3(query)
            
            evaluation_data.append({
                'Query': query,
                'Method': method_name,
                'Test1_Retrieved': len(test1_results),
                'Test2_Retrieved': len(test2_results),
                'Test3_Retrieved': len(test3_results)
            })
        else:
            # Run test 4 for advanced methods
            test4_results = searcher.search_test4(query)
            
            evaluation_data.append({
                'Query': query,
                'Method': method_name,
                'Test4_Retrieved': len(test4_results)
            })
    
    return pd.DataFrame(evaluation_data)

def generate_evaluation_table(baseline_results, method1_results, method2_results):
    """Generate the evaluation table for the report"""
    print("\n" + "="*100)
    print("EVALUATION RESULTS TABLE")
    print("="*100)
    
    queries = [
        "audio quality:poor",
        "wifi signal:strong",
        "mouse button:click problem", 
        "gps map:useful",
        "image quality:sharp"
    ]
    
    # Create table header
    print(f"{'Query':<30} | {'Baseline (Boolean)':<30} | {'Method 1 (M1)':<30} | {'Method 2 (M2)':<30}")
    print(f"{'':<30} | {'# Ret. # Rel. Prec.':<30} | {'# Ret. # Rel. Prec.':<30} | {'# Ret. # Rel. Prec.':<30}")
    print("-"*130)
    
    # Note: In a real implementation, you would need ground truth to calculate # Rel. and Precision
    # Here we're showing the structure
    for query in queries:
        baseline_ret = baseline_results[baseline_results['Query'] == query]['Test2_Retrieved'].values[0]
        method1_ret = method1_results[method1_results['Query'] == query]['Test4_Retrieved'].values[0]
        method2_ret = method2_results[method2_results['Query'] == query]['Test4_Retrieved'].values[0]
        
        # Placeholder values for relevant docs and precision
        # In practice, you'd calculate these against ground truth
        print(f"{query:<30} | {baseline_ret:<6} {'?':<6} {'?':<6} | {method1_ret:<6} {'?':<6} {'?':<6} | {method2_ret:<6} {'?':<6} {'?':<6}")
    
    print("\nNote: '?' indicates values that require ground truth labels to calculate")

def main():
    """Main pipeline execution"""
    print("Starting NLP Opinion Search Engine Pipeline")
    print("="*50)
    
    # Step 1: Create output directories
    create_output_directories()
    
    # Step 2: Preprocess data (only if not already done)
    if not os.path.exists('preprocessed_data.pkl'):
        print("\nStep 1: Preprocessing data...")
        preprocessor = DataPreprocessor('reviews_segment.pkl')
        df = preprocessor.load_data()
        df = preprocessor.preprocess_reviews()
        inverted_index = preprocessor.create_inverted_index()
        preprocessor.save_preprocessed_data()
    else:
        print("\nPreprocessed data already exists, skipping preprocessing...")
    
    # Define test queries
    test_queries = [
        "audio quality:poor",
        "wifi signal:strong",
        "mouse button:click problem",
        "gps map:useful", 
        "image quality:sharp"
    ]
    
    # Step 3: Run Baseline Boolean Search
    print("\nStep 2: Running Baseline Boolean Search...")
    baseline_searcher = BaselineBooleanSearch()
    baseline_results = run_baseline_searches(baseline_searcher, test_queries)
    baseline_eval = evaluate_results("Baseline", test_queries, baseline_searcher)
    
    # Step 4: Run Method 1 - Rating + Lexicon
    print("\nStep 3: Running Method 1 (Rating + Lexicon)...")
    method1_searcher = RatingLexiconSearch()
    method1_eval = evaluate_results("Method 1", test_queries, method1_searcher)
    
    # Save Method 1 results
    for query in test_queries:
        query_name = query.replace(" ", "_").replace(":", "_")
        results = method1_searcher.search_test4(query)
        method1_searcher.save_results(results, f"outputs/method1/{query_name}_test4.txt")
    
    # Step 5: Run Method 2 - Semantic Search
    print("\nStep 4: Running Method 2 (Semantic Search)...")
    method2_searcher = SimplifiedSemanticSearch()
    method2_eval = evaluate_results("Method 2", test_queries, method2_searcher)
    
    # Save Method 2 results (these go to advanced_models folder for final submission)
    for query in test_queries:
        query_name = query.replace(" ", "_").replace(":", "_")
        results = method2_searcher.search_test4(query)
        method2_searcher.save_results(results, f"outputs/advanced_models/{query_name}_test4.txt")
    
    # Step 6: Generate evaluation table
    generate_evaluation_table(baseline_eval, method1_eval, method2_eval)
    
    # Step 7: Summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print(f"  Baseline Test 2: {baseline_results[query]['test2']} documents")
        
        m1_count = method1_eval[method1_eval['Query'] == query]['Test4_Retrieved'].values[0]
        m2_count = method2_eval[method2_eval['Query'] == query]['Test4_Retrieved'].values[0]
        
        print(f"  Method 1 Test 4: {m1_count} documents")
        print(f"  Method 2 Test 4: {m2_count} documents")
    
    print("\n" + "="*50)
    print("Pipeline completed successfully!")
    print("Check the 'outputs' directory for all generated results")
    print("="*50)

if __name__ == "__main__":
    main()