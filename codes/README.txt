In python to read the file

import pandas as pd

# Read the DataFrame from the pickle file
df = pd.read_pickle("reviews_segment.pkl")

# Output the DataFrame first 5 columns
df.head()


Excel file
column names are provided as in the original .sql file, last column is th entire row of .sql file