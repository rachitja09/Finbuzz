import pandas as pd

def print_df_info(df: pd.DataFrame, name: str = "DataFrame"):
    print(f"\n=== {name} info ===")
    print(df.dtypes)
    print("\nHead:")
    print(df.head())
    print("\nObject columns:")
    for col in df.columns:
        if df[col].dtype == 'O':
            print(f"- {col}: {df[col].head().tolist()}")
    print("====================\n")

# Example usage:
# from utils.df_debug import print_df_info
# print_df_info(your_dataframe, "your_dataframe_name")
