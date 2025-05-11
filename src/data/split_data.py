import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(input_path, output_dir):
    # Load the raw data
    data = pd.read_csv(input_path)
    
    # Separate features and target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Save the splits
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

if __name__ == "__main__":
    import argparse
    import os

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Split raw data into training and testing sets.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the raw.csv file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the processed datasets.")
    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Call the split_data function
    split_data(args.input_path, args.output_dir)