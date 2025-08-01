import pandas as pd
import pickle

# Debug script to understand the data structure
print("Loading data to check structure...")
df = pd.read_pickle('reviews_segment.pkl')

print("\nDataFrame Info:")
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst few rows of key columns:")

# Display sample data
key_columns = ['review_id', 'customer_review_rating', 'review_text']
available_columns = [col for col in key_columns if col in df.columns]
print(df[available_columns].head())

# Check data types
print("\nData types:")
print(df.dtypes)

# Check rating distribution
if 'customer_review_rating' in df.columns:
    print("\nRating distribution:")
    print(df['customer_review_rating'].value_counts().sort_index())
    
# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Save a small sample for testing
print("\nSaving small sample for testing...")
sample_df = df.head(1000)
sample_df.to_pickle('sample_reviews.pkl')
print("Sample saved to sample_reviews.pkl")