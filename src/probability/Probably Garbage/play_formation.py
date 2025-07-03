import pandas as pd

# Load the dataset
data = pd.read_csv("formation_data_csv.csv")

# Define a function to infer formation
def infer_formation(row):
    # Step 1: Use explicit formation columns if available
    if pd.notna(row['pff_OFFFORMATION']):
        return row['pff_OFFFORMATION']
    
    # Step 2: Deduce formation based on personnel
    personnel = row['pff_OFFPERSONNELBASIC']
    shotgun = row['pff_SHOTGUN']
    pistol = row['pff_PISTOL']
    rbs_in_backfield = row['pff_RBSINBACKFIELD']
    
    if personnel == '2-0-2-1':
        if shotgun == 'S':
            return "Shotgun Split Back"
        elif pistol == 'P':
            return "Pistol"
        else:
            return "I-Formation"
    
    if personnel == '1-0-2-2':
        if shotgun == 'S':
            return "Shotgun"
        else:
            return "Singleback"
    
    # Add more rules based on known formation mappings
    if rbs_in_backfield == 0 and shotgun == 'S':
        return "Empty Backfield"
    
    # Step 3: Default to unknown if no rules match
    return "Unknown Formation"

# Apply the function to the dataset
data['InferredFormation'] = data.apply(infer_formation, axis=1)

# Save the results
data.to_csv("play_data_test.csv", index=False)

# Display a few rows
print(data[['pff_PLAYID', 'InferredFormation']].head())
