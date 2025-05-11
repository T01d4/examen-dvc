import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import dill
import os

def normalize_data(input_dir, output_dir, scaler_path):
    # Load the datasets
    X_train = pd.read_csv(f"{input_dir}/X_train.csv")
    X_test = pd.read_csv(f"{input_dir}/X_test.csv")
    
    # Identify numeric columns
    numeric_columns = X_train.select_dtypes(include=["number"]).columns
    
    # Normalize only numeric columns
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train[numeric_columns])
    X_test_scaled = scaler.transform(X_test[numeric_columns])
    
    # Combine scaled numeric columns with non-numeric columns
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=numeric_columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=numeric_columns, index=X_test.index)
    X_train_scaled = pd.concat([X_train_scaled, X_train.drop(columns=numeric_columns)], axis=1)
    X_test_scaled = pd.concat([X_test_scaled, X_test.drop(columns=numeric_columns)], axis=1)
    
    # Save the normalized datasets
    X_train_scaled.to_csv(f"{output_dir}/X_train_scaled.csv", index=False)
    X_test_scaled.to_csv(f"{output_dir}/X_test_scaled.csv", index=False)
    
    # Save the scaler
    with open(scaler_path, "wb") as f:
        dill.dump(scaler, f)

if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Normalize training and testing datasets.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input datasets.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save normalized datasets.")
    parser.add_argument("--scaler_path", type=str, required=True, help="Path to save the scaler.")
    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Call the normalize_data function
    normalize_data(args.input_dir, args.output_dir, args.scaler_path)