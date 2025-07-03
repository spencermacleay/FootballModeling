import pandas as pd

def calculate_proportions_and_replace(csv_file, output_file, original_data):
    # Read CSV files
    data = pd.read_csv(csv_file)
    data2 = pd.read_csv(original_data)

    # Columns to drop (only drop if they exist)
    columns_to_drop = [
        'offense', 'hash_detail', 'play_id', 
        'no_play', 'run_pass', 'pass_direction', 
        'run_pass_option', 'route_thrown'
    ]

    # Drop only columns that exist
    existing_columns_to_drop = [col for col in columns_to_drop if col in data.columns]
    data = data.drop(columns=existing_columns_to_drop, errors='ignore')
    data2 = data2.drop(columns=existing_columns_to_drop, errors='ignore')

    # Remove rows where 'adj_run_pass' is 'X' only if the column exists
    if "adj_run_pass" in data.columns:
        data = data[data["adj_run_pass"] != "X"]
    if "adj_run_pass" in data2.columns:
        data2 = data2[data2["adj_run_pass"] != "X"]

    # Identify numerical and categorical columns
    numerical_columns = data.select_dtypes(include=['number']).columns
    categorical_columns = [col for col in ["tight_ends", "formation_in_boundary", "off_personnel_group", "backset", "running_backs", "tight_ends", "wide_receivers", 'pass_concept_half_field_left', 'pass_concept_half_field_right', 'pass_concept_half_field_special'] if col in data.columns]

    # Ensure only existing categorical columns are processed
    proportion_lookup = {}
    for col in categorical_columns:
        total_count = len(data[col])
        value_counts = data[col].value_counts(dropna=False).to_dict()
        value_probs = {k: v / total_count for k, v in value_counts.items()}
        proportion_lookup[col] = value_probs

    # Replace values only if the column exists
    for col, mapping in proportion_lookup.items():
        if col in data.columns:
            data[col] = data[col].map(mapping)
        if col in data2.columns:
            data2[col] = data2[col].map(mapping)

    # Define target columns for One-Hot Encoding
    target_categorical_columns = ["primary_run_concept", "secondary_run_concept", "pass_concept_targeted"]

    # **Ensure "NONE" category exists before encoding**
    for col in target_categorical_columns:
        if col in data.columns:
            data[col] = data[col].fillna("NONE")  # Fill missing concepts with "NONE"
        if col in data2.columns:
            data2[col] = data2[col].fillna("NONE")  # Ensure consistency in both datasets

    # Apply One-Hot Encoding including "NONE"
    data = pd.get_dummies(data, columns=target_categorical_columns, dtype=int)
    data2 = pd.get_dummies(data2, columns=target_categorical_columns, dtype=int)

    # Ensure data2 has the same columns as data (fill missing with 0)
    data2 = data2.reindex(columns=data.columns, fill_value=0)

    # **Fill NaN values with 0 after one-hot encoding**
    data = data.fillna(0)
    data2 = data2.fillna(0)

    # Ensure gain_loss is numeric
    if 'gain_loss' in data.columns:
        data["gain_loss"] = pd.to_numeric(data["gain_loss"], errors='coerce').fillna(0)
    if 'gain_loss' in data2.columns:
        data2["gain_loss"] = pd.to_numeric(data2["gain_loss"], errors='coerce').fillna(0)

    # Ensure adj_run_pass is mapped to numeric
    if 'adj_run_pass' in data.columns:
        data['adj_run_pass'] = data['adj_run_pass'].map({'R': 0, 'P': 1}).fillna(0).astype(int)
    if 'adj_run_pass' in data2.columns:
        data2['adj_run_pass'] = data2['adj_run_pass'].map({'R': 0, 'P': 1}).fillna(0).astype(int)

    # Drop empty columns after all processing
    data = data.dropna(axis=1, how="all")
    data2 = data2.dropna(axis=1, how="all")

    # **Check if any string values remain**
    non_numeric_columns = data.select_dtypes(include=['object']).columns
    if len(non_numeric_columns) > 0:
        print(f"⚠️ WARNING: Non-numeric columns detected: {list(non_numeric_columns)}")
    else:
        print("✅ All columns are numeric.")

    # Save modified CSV while keeping only existing headers
    data.to_csv(output_file, index=False)
    data2.to_csv(original_data, index=False)

    print(f"Modified CSV with One-Hot Encoded targets saved to {output_file}")

# Main Execution
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python calculate_proportions.py <input_csv> <output_csv> <original_csv>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    original_csv = sys.argv[3]

    calculate_proportions_and_replace(input_csv, output_csv, original_csv)
