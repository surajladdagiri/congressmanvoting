import pandas as pd

# Read the original CSV into a DataFrame
df = pd.read_csv("billinfo.csv")

# Get the unique congress values in the DataFrame
unique_congress = df['congress'].unique()

# Loop over each unique congress value and save the subset to a separate file
for congress_number in unique_congress:
    subset_df = df[df['congress'] == congress_number]
    subset_df.to_csv(f"{congress_number}.csv", index=False)