import pandas as pd
import numpy as np
import shutil
import sys

# entire_dataset_columns = [
#     "PLAYID", "GAMEID", "GAMEDATE", "GAMESEASON", "WEEK", "GSISGAMEKEY", "GSISPLAYID", "QUARTER", "DOWN", 
#     "CLOCK", "BLITZDOG", "CATCHABLE", "DEEPPASS", "DEFSCORE", "DRAW", "DRIVE", "DRIVEPLAY", "GAINLOSS", 
#     "GARBAGETIME", "FIRST_DOWN_GAINED", "GETOFFTIME", "KICKYARDS", "NOHUDDLE", "NOPLAY", 
#     "OFFFORMATIONUNBALANCED", "OFFSCORE", "OPTION", "PENALTYYARDS", "PLAYACTION", "PREVIOUSPFFPLAYID", 
#     "NEXTPFFPLAYID", "RETURNYARDS", "SCORE", "SCREEN", "SNAPTIME", "SORTORDER", "STUNT", 
#     "TIMETOPRESSURE", "TIMETOTHROW", "TRICKLOOK", "TRICKPLAY", "YARDSAFTERCATCH", "YARDSAFTERCONTACT", 
#     "2MINUTE", "3MINUTE", "BALLCARRIER", "BOXPLAYERS", "BUNCHED", "CENTERPASSBLOCKDIRECTION", 
#     "CHECKROUTE", "CHIPROUTE", "DBDEPTH", "DEFFRONT", "DEFPERSONNEL", "DEFPLAYERS", 
#     "DEFPLAYERSRATINGS", "DEFSUBSTITUTIONS", "DEFSUCCESS", "DEFTEAM", "DISTANCE", "DLDROP", 
#     "DLTECHNIQUES", "DOUBLETEAM", "DRIVEENDEVENT", "DRIVEENDFIELDPOSITION", "DRIVEENDPLAYNUMBER", 
#     "DRIVESTARTEVENT", "DRIVESTARTFIELDPOSITION", "DROPBACKDEPTH", "DROPBACKTYPE", "FIELDPOSITION", 
#     "FIRSTCONTACT", "FORCEDFUMBLE", "FUMBLE", "FUMBLERECOVERY", "GAINLOSSNET", "GUNNERS", "HANGTIME", 
#     "HASH", "HASHDEF", "HIT", "HURRY", "INCOMPLETIONTYPE", "INJURED", "INTERCEPTION", "KEYPLAYERS", 
#     "KICKCONTACT", "KICKDEPTH", "KICKDIRACTUAL", "KICKDIRINTENDED", "KICKER", "KICKRESULT", "KICKTYPE", 
#     "KICKWIDTH", "KICKZONE", "LBDEPTH", "MISSEDTACKLE", "MOFOCPLAYED", "MOFOCSHOWN", "NEGATIVEPFFGRADE", 
#     "OFFFORMATION", "OFFFORMATIONGROUP", "OFFODDITIES", "OFFPERSONNEL", "OFFPERSONNELBASIC", 
#     "OFFPERSONNELSKILL", "OFFPLAYERS", "OFFPLAYERSRATINGS", "OFFSUBSTITUTIONS", "OFFSUCCESS", 
#     "OFFTEAM", "ONLOS", "OPERATIONTIME", "PASSBLOCKING", "PASSBREAKUP", "PASSCOVERAGE", 
#     "PASSCOVERAGE1", "PASSCOVERAGE2", "PASSCOVERAGEPLAYERS", "PASSDEPTH", "PASSDIRECTION", "PASSER", 
#     "PASSPATTERN", "PASSPATTERNBASIC", "PASSPATTERNBYPLAYER", "PASSRECEIVERPOSITIONTARGET", 
#     "PASSRECEIVERTARGET", "PASSRESULT", "PASSROUTETARGET", "PASSROUTETARGETGROUP", "PASSRUSHPLAYERS", 
#     "PASSRUSHRESULT", "PASSWIDTH", "PASSZONE", "PENALTY", "PISTOL", "PLAYACTIONFAKE", 
#     "PLAYENDFIELDPOSITION", "POAACTUAL", "POACHANGEREASON", "POAINTENDED", "POSITIVEPFFGRADE", "PRESS", 
#     "PRESSUREDETAIL", "PUMPFAKE", "PUNTRUSH", "PURSUIT", "QB", "QBMOVEDOFFSPOT", "QBPRESSURE", 
#     "QBPRESSUREALLOWED", "QBRESET", "QBSCRAMBLE", "RBALIGNMENT", "RBDEPTH", "RBDIRECTION", 
#     "RBSINBACKFIELD", "RETDIRECTIONINTENDED", "RETURNDIRECTION", "RETURNER", "RUNCONCEPTPRIMARY", 
#     "RUNCONCEPTSECONDARY", "RUNPASS", "SACK", "SCOREDIFFERENTIAL", "SHIFTMOTION", "SHOTGUN", 
#     "SPECIALTEAMSTYPE", "STOP", "STSAFETIES", "TACKLE", "TACKLEASSIST", "TEALIGNMENT", "TOUCHDOWN", 
#     "UNBLOCKEDPRESSURE", "VISE", "WRALIGNMENT", "CONTESTED", "RUNPASSOPTION", "PLAYCLOCK"]

# Transform the FIELDPOSITION column to yards_to_goal_line
def transform_field_position(field_position):
    if field_position < 0:  # Negative: On own side
        return 100 + field_position  # Convert to positive
    else:  # Positive: On opponent's side
        return 100 - field_position

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

def process_column(df, column):
    if column in df.columns:
        df[column] = df[column].astype(str).fillna("MISSING")  # Convert all values to string & handle NaN
        
        # Count occurrences of each unique value
        value_counts = df[column].value_counts().sort_index()  # Ensure sorted order for cumulative sum
        total_sum = value_counts.sum()  # Get total count of all values
        cum_sum = value_counts.cumsum()  # Compute cumulative sum
        
        # Compute cumulative probability
        cumulative_probability = (cum_sum / total_sum).to_dict()
        
        # ✅ Use `.replace()` instead of `.map()`
        df[column] = df[column].replace(cumulative_probability)


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

def preprocess_data(input_csv, output_csv):
    """Preprocesses play data for machine learning."""
    
    # Load data
    df = pd.read_csv(input_csv)
    df.columns = df.columns.str.replace('^pff_', '', regex=True)

    # 🔻 Drop unnecessary columns if they exist
    drop_columns = [
    "PLAYID", "ONLOS", "GAMEID", "GAMEDATE", "GAMESEASON", "WEEK", "GSISGAMEKEY", "GSISPLAYID",
    "CLOCK", "BLITZDOG", "CATCHABLE", "DEEPPASS", "DEFSCORE", "DRAW", "DRIVE", "DRIVEPLAY",
    "GARBAGETIME", "FIRST_DOWN_GAINED", "GETOFFTIME", "KICKYARDS", "NOHUDDLE", "OFFSCORE", 
    "OPTION", "PENALTYYARDS", "PLAYACTION", "PREVIOUSPFFPLAYID", "NEXTPFFPLAYID", "RETURNYARDS", "SCREEN", "SNAPTIME", "SORTORDER",
    "STUNT", "TIMETOPRESSURE", "TIMETOTHROW", "TRICKLOOK", "TRICKPLAY", "YARDSAFTERCATCH",
    "YARDSAFTERCONTACT", "3MINUTE", "BALLCARRIER", "BOXPLAYERS", "BUNCHED",
    "CENTERPASSBLOCKDIRECTION", "CHECKROUTE", "CHIPROUTE", "DEFFRONT",
    "DEFPERSONNEL", "DEFPLAYERS", "DEFPLAYERSRATINGS", "DEFSUBSTITUTIONS", "DEFSUCCESS",
    "DEFTEAM", "DLDROP", "DLTECHNIQUES", "DOUBLETEAM", "DRIVEENDEVENT", "DRIVEENDFIELDPOSITION",
    "DRIVEENDPLAYNUMBER", "DRIVESTARTEVENT", "DRIVESTARTFIELDPOSITION", "DROPBACKDEPTH",
    "DROPBACKTYPE", "FIRSTCONTACT", "FORCEDFUMBLE", "FUMBLE", "FUMBLERECOVERY", "GAINLOSSNET",
    "GUNNERS", "HANGTIME", "HASHDEF", "HIT", "HURRY", "INCOMPLETIONTYPE", "INJURED",
    "INTERCEPTION", "KEYPLAYERS", "KICKCONTACT", "KICKDEPTH", "KICKDIRACTUAL", "KICKDIRINTENDED",
    "KICKER", "KICKRESULT", "KICKTYPE", "KICKWIDTH", "KICKZONE", "MISSEDTACKLE",
    "MOFOCPLAYED", "MOFOCSHOWN", "NEGATIVEPFFGRADE", "OFFPERSONNELBASIC", "OFFPERSONNELSKILL", "OFFPLAYERS",
    "OFFPLAYERSRATINGS", "OFFSUBSTITUTIONS", "OFFSUCCESS", "OPERATIONTIME",
    "PASSBLOCKING", "PASSBREAKUP", "PASSCOVERAGE", "PASSCOVERAGE1", "PASSCOVERAGE2",
    "PASSCOVERAGEPLAYERS", "PASSDEPTH", "PASSDIRECTION", "PASSER", "PASSPATTERN",
    "PASSPATTERNBASIC", "PASSPATTERNBYPLAYER", "PASSRECEIVERPOSITIONTARGET",
    "PASSRECEIVERTARGET", "PASSRESULT", "PASSROUTETARGET", "PASSROUTETARGETGROUP",
    "PASSRUSHPLAYERS", "PASSRUSHRESULT", "PASSWIDTH", "PASSZONE", "PENALTY",
    "PLAYACTIONFAKE", "PLAYENDFIELDPOSITION", "POAACTUAL", "POACHANGEREASON", "POAINTENDED",
    "POSITIVEPFFGRADE", "PRESS", "PRESSUREDETAIL", "PUMPFAKE", "PUNTRUSH", "PURSUIT", "QB",
    "QBMOVEDOFFSPOT", "QBPRESSURE", "QBPRESSUREALLOWED", "QBRESET", "QBSCRAMBLE", 
    "RBDIRECTION", "RBSINBACKFIELD", "RETDIRECTIONINTENDED",
    "RETURNDIRECTION", "RETURNER", "SACK", "SHIFTMOTION", "STOP", "STSAFETIES", "TACKLE", 
    "TACKLEASSIST", "TEALIGNMENT", "TOUCHDOWN", "UNBLOCKEDPRESSURE", "VISE", "CONTESTED", 
    "PLAYCLOCK", "RUNCONCEPTPRIMARY", "RUNCONCEPTSECONDARY", "HASH",
    "RUNPASSOPTION", "index",
    ]

    df.drop(columns=[col for col in drop_columns if col in df], inplace=True, errors="ignore")
    # 🎯 **Feature Engineering**

    df = df[df["NOPLAY"] == 0]  # Keep only valid plays
    df = df[df["SPECIALTEAMSTYPE"].isna()]

    df["weighted_down_distance"] = df["DISTANCE"] * df["DOWN"].apply(lambda x: x if x in [1, 2, 3, 4] else 0)
    
    # Conversion into yards_to_goal equivalent
    df['yards_to_goal_line'] = df['FIELDPOSITION'].apply(transform_field_position)
    df.reset_index(inplace=True)

    df = df.sort_values(by=["GameID", "Team", "PlayNumber"])  # Ensure correct order
    df["Prev_Play_Yards"] = df.groupby(["GameID", "Team"])["YardsGained"].shift(1)
    df["Prev_Play_Yards"].fillna(0, inplace=True)  # Assume 0 for first play

    ## **1️⃣ Convert `yards_to_goal_line` into bins**
    yard_bins = list(range(0, 110, 10))  # 0-10, 10-20, ..., 90-100
    df["yards_to_goal_bin"] = pd.cut(df["yards_to_goal_line"], bins=yard_bins, labels=False)
    
    df.drop(columns=["yards_to_goal_line"], inplace=True)

    ## **2️⃣ Normalize `win_prob` from `score_differential`**
    df["win_prob"] = logistic_win_prob(df["SCOREDIFFERENTIAL"]).clip(-1, 1)

    df.drop(columns=["SCOREDIFFERENTIAL", "SCORE"], inplace=True)

    # # Ensure no NaN values in the column
    # df["OFFFORMATIONGROUP"].fillna("0x0", inplace=True)

    # Columns to process
    columns_to_process = ['OFFODDITIES', 'OFFPERSONNEL', 'PISTOL', 'RBALIGNMENT', 'RBDEPTH', 'SHOTGUN', 
        'WRALIGNMENT', "OFFFORMATION", "DOWN", 'DBDEPTH', 'LBDEPTH', 'OFFFORMATIONGROUP']
    for column in columns_to_process:
        df[column].fillna("N/A", inplace=True)

    # Output unique values for inspection before processing
    for column in columns_to_process:
        if column in df.columns:
            unique_values = df[column].unique()
            print(f"Unique values in column '{column}': {unique_values}")
            process_column(df, column)

    # 'DBDEPTH', 'LBDEPTH', 'OFFODDITIES', 'OFFPERSONNEL', 'ONLOS', 'PISTOL', 'RBALIGNMENT', 'RBDEPTH', 'SHOTGUN', 'WRALIGNMENT'
    # categorical_features = ["DOWN"]
    # df = pd.get_dummies(df, columns=categorical_features, dtype=int)

    ## **7️⃣ Encode `adj_run_pass`**
    df["RUNPASS"] = df["RUNPASS"].map({"R": 0, "P": 1}).fillna(0).astype(int)
    
    # Calculate Run Rate per Team
    team_run_rate = df.groupby("OFFTEAM")["RUNPASS"].mean().to_dict()  # 0 = Run, 1 = Pass

    # Add the Run Rate to the Dataset
    df["TEAM_RUN_RATE"] = df["OFFTEAM"].map(team_run_rate)

    ## **8️⃣ Final Cleanup**
    df.fillna(0, inplace=True)

    # Ensure all columns are numeric
    non_numeric_columns = df.select_dtypes(include=["object"]).columns
    if len(non_numeric_columns) > 0:
        print(f"⚠️ WARNING: Non-numeric columns detected: {list(non_numeric_columns)}")
    else:
        print("✅ All columns are numeric.")

    # Save processed files
    df.to_csv(output_csv, index=False)

    print(f"✅ Processed data saved to {output_csv}")

### **🚀 Main Execution**
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python preprocess.py <input_csv> <output_csv> <original_data")
        sys.exit(1)
    # python3 attemptOnLargeData.py play_feed_ncaa.csv processed_play_feed_ncaa.csv copy_play_feed_ncaa.csv

    input_csv, output_csv, original_csv = sys.argv[1:4]

    # Ensure reference dataset is always fresh
    shutil.copy(original_csv, input_csv)

    preprocess_data(input_csv, output_csv)
