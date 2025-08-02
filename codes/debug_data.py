import pandas as pd
import pickle

print("Loading data to check structure...")
df = pd.read_pickle('reviews_segment.pkl')

print("\nDataFrame Info:")
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst few rows of key columns:")

key_columns = ['review_id', 'customer_review_rating', 'review_text']
available_columns = [col for col in key_columns if col in df.columns]
print(df[available_columns].head())

print("\nData types:")
print(df.dtypes)

if 'customer_review_rating' in df.columns:
    print("\nRating distribution:")
    print(df['customer_review_rating'].value_counts().sort_index())
    
print("\nMissing values:")
print(df.isnull().sum())

print("\nSaving small sample for testing...")
sample_df = df.head(1000)
sample_df.to_pickle('sample_reviews.pkl')
print("Sample saved to sample_reviews.pkl")