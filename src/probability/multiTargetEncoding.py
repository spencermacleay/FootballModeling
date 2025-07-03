import pandas as pd
import numpy as np
import shutil
import sys

### ⚡ Utility Functions ###

def logistic_win_prob(score_diff):
    """Applies a sigmoid-like transformation to score differential for win probability."""
    return 2 / (1 + np.exp(-score_diff / 10)) - 1  # Sigmoid-like scaling

def simplify_backset(backset):
    """Group backset formations into fewer categories."""
    if pd.isna(backset):
        return "other"
    if "GUN" in backset:
        return "shotgun"
    if "PISTOL" in backset:
        return "pistol"
    if "I" in backset:
        return "under_center"
    if "VICTORY" in backset:
        return "victory"
    return "other"

def simplify_personnel(personnel):
    """Group personnel types into common categories."""
    if pd.isna(personnel):
        return "other"
    if "10" in personnel:
        return "spread"
    if "11" in personnel:
        return "balanced"
    if "12" in personnel or "13" in personnel:
        return "heavy"
    if "21" in personnel or "22" in personnel:
        return "power"
    return "other"

### 🔄 Main Processing Function ###

def preprocess_data(input_csv, output_csv, ref_csv):
    """Preprocesses play data for machine learning."""
    
    # Load data
    df_main = pd.read_csv(input_csv)
    df_ref = pd.read_csv(ref_csv)

    # 🔻 Drop unnecessary columns if they exist
    drop_columns = [
        "offense", "hash_detail", "play_id", "no_play", "run_pass", "pass_direction",
        "run_pass_option", "route_thrown", "pass_concept_half_field_left",
        "pass_concept_half_field_right", "pass_concept_half_field_special",
        "garbage_time", "two_minute", "formation_in_boundary", "field_position",
        "quarter", "clock", "quarterback_number", "gain_loss"
    ]
    df_main.drop(columns=[col for col in drop_columns if col in df_main], inplace=True, errors="ignore")
    df_ref.drop(columns=[col for col in drop_columns if col in df_ref], inplace=True, errors="ignore")

    # 🎯 **Feature Engineering**
    
    ## **1️⃣ Convert `yards_to_goal_line` into bins**
    yard_bins = list(range(0, 110, 10))  # 0-10, 10-20, ..., 90-100
    df_main["yards_to_goal_bin"] = pd.cut(df_main["yards_to_goal_line"], bins=yard_bins, labels=False)
    df_ref["yards_to_goal_bin"] = pd.cut(df_ref["yards_to_goal_line"], bins=yard_bins, labels=False)
    
    df_main.drop(columns=["yards_to_goal_line"], inplace=True)
    df_ref.drop(columns=["yards_to_goal_line"], inplace=True)

    ## **2️⃣ Normalize `win_prob` from `score_differential`**
    df_main["win_prob"] = logistic_win_prob(df_main["score_differential"]).clip(-0.5, 0.5)
    df_ref["win_prob"] = logistic_win_prob(df_ref["score_differential"]).clip(-0.5, 0.5)

    df_main.drop(columns=["score_differential", "score"], inplace=True)
    df_ref.drop(columns=["score_differential", "score"], inplace=True)

    # Replace personnel frequency with relative importance
    for df in [df_main, df_ref]:
        if "backset" in df.columns:
            df["backset_type"] = df["backset"].apply(simplify_backset)  # Convert to general categories
            df["backset_type"] = df["backset_type"].astype("category").cat.codes  # Encode numerically
            df.drop(columns=["backset"], inplace=True)  # Drop original column

        if "off_personnel_group" in df.columns:
            # Compute Personnel Frequency
            personnel_counts = df["off_personnel_group"].value_counts(normalize=True).to_dict()
            df["personnel_frequency"] = df["off_personnel_group"].map(personnel_counts)

            # Convert personnel_frequency into a binary feature (high vs. low usage)
            median_freq = df["personnel_frequency"].median()
            df["personnel_frequency_high"] = (df["personnel_frequency"] >= median_freq).astype(int)

            # ✅ Compute Interaction Before Dropping personnel_frequency
            if "spread_index" in df.columns:
                df["personnel_spread_interaction"] = df["personnel_frequency"] * df["spread_index"]

            # Drop the original continuous personnel_frequency column
            df.drop(columns=["personnel_frequency"], inplace=True)

            # Drop original categorical personnel column
            df.drop(columns=["off_personnel_group"], inplace=True, errors="ignore")

    

    ## **4️⃣ Process Numerical Features (WR, RB, TE counts)**
    for df in [df_main, df_ref]:
        df["num_wide_receivers"] = df["wide_receivers"].apply(lambda x: len(str(x).split(";")))
        df["num_running_backs"] = df["running_backs"].apply(lambda x: len(str(x).split(";")))
        df["num_tight_ends"] = df["tight_ends"].apply(lambda x: len(str(x).split(";")))

        df["spread_index"] = df["num_wide_receivers"] - df["num_tight_ends"] - df["num_running_backs"]


        df.drop(columns=["wide_receivers", "running_backs", "tight_ends", "num_wide_receivers", "num_tight_ends", "num_running_backs"], inplace=True)

    ## **5️⃣ Encode Play Call Targets (One-Hot Encoding)**
    categorical_features = ["primary_run_concept", "secondary_run_concept", "pass_concept_targeted", "down"]
    df_main = pd.get_dummies(df_main, columns=categorical_features, dtype=int)
    df_ref = pd.get_dummies(df_ref, columns=categorical_features, dtype=int)

    ## **6️⃣ Feature Engineering: Distance x Down**
    for df in [df_main, df_ref]:
        df["distance_x_down"] = df["distance"] * (df["down_1"] + 2 * df["down_2"] + 3 * df["down_3"] + 4 * df["down_4"])
        df.drop(columns=["distance", "down_1", "down_2"], inplace=True)

    ## **7️⃣ Encode `adj_run_pass`**
    for df in [df_main, df_ref]:
        if "adj_run_pass" in df.columns:
            df["adj_run_pass"] = df["adj_run_pass"].map({"R": 0, "P": 1}).fillna(0).astype(int)

    ## **8️⃣ Final Cleanup**
    df.fillna(0, inplace=True)
    # Ensure all columns are numeric
    non_numeric_columns = df_main.select_dtypes(include=["object"]).columns
    if len(non_numeric_columns) > 0:
        print(f"⚠️ WARNING: Non-numeric columns detected: {list(non_numeric_columns)}")
    else:
        print("✅ All columns are numeric.")

    # Save processed files
    df_main.to_csv(output_csv, index=False)
    df_ref.to_csv(ref_csv, index=False)

    print(f"✅ Processed data saved to {output_csv}")

### **🚀 Main Execution**
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python preprocess.py <input_csv> <output_csv> <ref_csv> <unedited_csv>")
        sys.exit(1)

    input_csv, output_csv, ref_csv, unedited_csv = sys.argv[1:5]

    # Ensure reference dataset is always fresh
    shutil.copy(unedited_csv, ref_csv)

    preprocess_data(input_csv, output_csv, ref_csv)
