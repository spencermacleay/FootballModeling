'''
Copyright clemenmo 2025
Simple API-like program for communicating with Javascript.
'''

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from joblib import load

formationDict = {
    "LWR^; SLWR; HB-R; SRWR; RWR^": 0.040941,
    "LWR^; SLWR; HB-L; SRWR; RWR^": 0.037811,
    "LWR^; HB-R; TE-R; SRWR; RWR^": 0.028927,
    "LWR^; TE-L; HB-R; SRWR; RWR^": 0.024272,
    "LWR^; SLWR; TE-L; HB-L; RWR^": 0.023409,
    "LWR^; HB-L; TE-R; SRWR; RWR^": 0.023234,
    "LWR^; SLWR; HB-R; TE-R; RWR^": 0.023104,
    "LWR^; SLWR; HB-L; TE-R; RWR^": 0.022807,
    "LWR^; TE-L; HB-L; SRWR; RWR^": 0.020968,
    "LWR^; HB-L; SRiWR; SRoWR; RWR^": 0.020288
}

# Function to convert clock times to seconds
def timeToSecs(t):
    times = t.split(':')
    mins = int(times[0]) * 60
    secs = int(times[1])
    return mins + secs


def createDataframe(values):
    '''array structure:
    0: quarter
    1: down
    2: distance
    3: clock minutes
    4: clock seconds
    5: offensive score
    6: defensive score
    7: hash
    8: offensive personnel
    9: offensive formation
    10: field position

    needs to be:
    hash
    offscore
    defscore
    offformation
    offpersonnel
    quarter
    distance
    down
    clock
    fieldposition
    gainloss (7)
    '''
    # Put values in right order for prediction
    newList = list()
    clock = str(values[3]) + ":" + str(values[4])
    newList.extend([values[7], values[5], values[6], values[9], values[8], values[0], values[2], values[1], clock, values[10], 7])
    # Add col names
    column_names = [
    'pff_HASH',
    'pff_OFFSCORE',
    'pff_DEFSCORE',
    'pff_OFFFORMATION',
    'pff_OFFPERSONNELBASIC',
    'pff_QUARTER',
    'pff_DISTANCE',
    'pff_DOWN',
    'pff_CLOCK',
    'pff_FIELDPOSITION',
    'pff_GAINLOSS'
    ]
    values_df = pd.DataFrame([newList], columns=column_names)

    values_df['pff_CLOCK'] = values_df['pff_CLOCK'].apply(timeToSecs)
    values_df['pff_OFFFORMATION'] = values_df['pff_OFFFORMATION'].map(formationDict)
    values_df['pff_FIELDPOSITION'] = values_df['pff_FIELDPOSITION'].astype(int)
    values_df['pff_QUARTER'] = values_df['pff_QUARTER'].astype(int)
    values_df['pff_OFFSCORE'] = values_df['pff_OFFSCORE'].astype(int)
    values_df['pff_DEFSCORE'] = values_df['pff_DEFSCORE'].astype(int)
    values_df['pff_DOWN'] = values_df['pff_DOWN'].astype(int)
    values_df['pff_DISTANCE'] = values_df['pff_DISTANCE'].astype(int)
    values_df['pff_OFFPERSONNELBASIC'] = values_df['pff_OFFPERSONNELBASIC'].astype(int)

    # Add all feature engineered columns
    values_df['pff_HASH'] = values_df['pff_HASH'].map({'R': 0, 'L': 1, 'C': 2})

    values_df['fpos_boxed'] = values_df['pff_FIELDPOSITION'].between(0,10).astype(int)
    values_df['fpos_own_territory'] = values_df['pff_FIELDPOSITION'].between(11,44).astype(int)
    values_df['fpos_shots'] = values_df['pff_FIELDPOSITION'].between(45,65).astype(int)
    values_df['fpos_high_red'] = values_df['pff_FIELDPOSITION'].between(66,78).astype(int)
    values_df['fpos_red_zone'] = values_df['pff_FIELDPOSITION'].between(79,90).astype(int)
    values_df['fpos_low_red'] = values_df['pff_FIELDPOSITION'].between(91,99).astype(int)
    values_df['1st_10'] = ((values_df['pff_DOWN'] == 1) & (values_df['pff_DISTANCE'] == 10)).astype(int)
    values_df['1st_11+'] = ((values_df['pff_DOWN'] == 1) & (values_df['pff_DISTANCE'].between(11, 999))).astype(int)
    values_df['1st_9in'] = ((values_df['pff_DOWN'] == 1) & (values_df['pff_DISTANCE'].between(1, 9))).astype(int)
    values_df['2nd_11+'] = ((values_df['pff_DOWN'] == 2) & (values_df['pff_DISTANCE'].between(11, 999))).astype(int)
    values_df['2nd_7-10'] = ((values_df['pff_DOWN'] == 2) & (values_df['pff_DISTANCE'].between(7, 10))).astype(int)
    values_df['2nd_3-6'] = ((values_df['pff_DOWN'] == 2) & (values_df['pff_DISTANCE'].between(3, 6))).astype(int)
    values_df['2nd_1-2'] = ((values_df['pff_DOWN'] == 2) & (values_df['pff_DISTANCE'].between(1, 2))).astype(int)
    values_df['3rd_11+'] = ((values_df['pff_DOWN'] == 3) & (values_df['pff_DISTANCE'].between(11, 999))).astype(int)
    values_df['3rd_7-10'] = ((values_df['pff_DOWN'] == 3) & (values_df['pff_DISTANCE'].between(7, 10))).astype(int)
    values_df['3rd_6-4'] = ((values_df['pff_DOWN'] == 3) & (values_df['pff_DISTANCE'].between(6, 4))).astype(int)
    values_df['3rd_1-3'] = ((values_df['pff_DOWN'] == 3) & (values_df['pff_DISTANCE'].between(1, 3))).astype(int)
    values_df['4th_11+'] = ((values_df['pff_DOWN'] == 4) & (values_df['pff_DISTANCE'].between(11, 999))).astype(int)
    values_df['4th_6-10'] = ((values_df['pff_DOWN'] == 4) & (values_df['pff_DISTANCE'].between(6, 10))).astype(int)
    values_df['4th_3-5'] = ((values_df['pff_DOWN'] == 4) & (values_df['pff_DISTANCE'].between(3, 5))).astype(int)
    values_df['4th_1-2'] = ((values_df['pff_DOWN'] == 4) & (values_df['pff_DISTANCE'].between(1, 2))).astype(int)

    values_df['Q1_11:00-15:00'] = ((values_df['pff_QUARTER'] == 1) & (values_df['pff_CLOCK'].between(timeToSecs("11:00"), timeToSecs("15:00")))).astype(int)
    values_df['Q1_7:00-10:59'] = ((values_df['pff_QUARTER'] == 1) & (values_df['pff_CLOCK'].between(timeToSecs("11:00"), timeToSecs("15:00")))).astype(int)
    values_df['Q1_3:00-6:59'] = ((values_df['pff_QUARTER'] == 1) & values_df['pff_CLOCK'].between(timeToSecs("3:00"), timeToSecs("6:59"))).astype(int)
    values_df['Q1_0-2:59'] = ((values_df['pff_QUARTER'] == 1) & values_df['pff_CLOCK'].between(timeToSecs("0:00"), timeToSecs("2:59"))).astype(int)

    values_df['Q2_6-15'] = ((values_df['pff_QUARTER'] == 2) & values_df['pff_CLOCK'].between(timeToSecs("6:00"), timeToSecs("15:00"))).astype(int)
    values_df['Q2_4-6:59'] = ((values_df['pff_QUARTER'] == 2) & values_df['pff_CLOCK'].between(timeToSecs("4:00"), timeToSecs("6:59"))).astype(int)
    values_df['Q2_2-3:59'] = ((values_df['pff_QUARTER'] == 2) & values_df['pff_CLOCK'].between(timeToSecs("2:00"), timeToSecs("3:59"))).astype(int)
    values_df['Q2_1-1:59'] = ((values_df['pff_QUARTER'] == 2) & values_df['pff_CLOCK'].between(timeToSecs("1:00"), timeToSecs("1:59"))).astype(int)
    values_df['Q2_0:40-0:59'] = ((values_df['pff_QUARTER'] == 2) & values_df['pff_CLOCK'].between(timeToSecs("0:40"), timeToSecs("0:59"))).astype(int)
    values_df['Q2_0:20-0:39'] = ((values_df['pff_QUARTER'] == 2) & values_df['pff_CLOCK'].between(timeToSecs("0:20"), timeToSecs("0:39"))).astype(int)
    values_df['Q2_0:08-0:19'] = ((values_df['pff_QUARTER'] == 2) & values_df['pff_CLOCK'].between(timeToSecs("0:08"), timeToSecs("0:19"))).astype(int)
    values_df['Q2_0:00-0:07'] = ((values_df['pff_QUARTER'] == 2) & values_df['pff_CLOCK'].between(timeToSecs("0:00"), timeToSecs("0:07"))).astype(int)

    values_df['Q3_11-15'] = ((values_df['pff_QUARTER'] == 3) & values_df['pff_CLOCK'].between(timeToSecs("11:00"), timeToSecs("15:00"))).astype(int)
    values_df['Q3_7-10:59'] = ((values_df['pff_QUARTER'] == 3) & values_df['pff_CLOCK'].between(timeToSecs("7:00"), timeToSecs("10:59"))).astype(int)
    values_df['Q3_3-6:59'] = ((values_df['pff_QUARTER'] == 3) & values_df['pff_CLOCK'].between(timeToSecs("3:00"), timeToSecs("6:59"))).astype(int)
    values_df['Q3_0-2:59'] = ((values_df['pff_QUARTER'] == 3) & values_df['pff_CLOCK'].between(timeToSecs("0:00"), timeToSecs("2:59"))).astype(int)

    values_df['Q4_7-15'] = ((values_df['pff_QUARTER'] == 4) & values_df['pff_CLOCK'].between(timeToSecs("7:00"), timeToSecs("15:00"))).astype(int)
    values_df['Q4_5:30-6:59'] = ((values_df['pff_QUARTER'] == 4) & values_df['pff_CLOCK'].between(timeToSecs("5:30"), timeToSecs("6:59"))).astype(int)
    values_df['Q4_2-5:29'] = ((values_df['pff_QUARTER'] == 4) & values_df['pff_CLOCK'].between(timeToSecs("2:00"), timeToSecs("5:29"))).astype(int)
    values_df['Q4_1-1:59'] = ((values_df['pff_QUARTER'] == 4) & values_df['pff_CLOCK'].between(timeToSecs("1:00"), timeToSecs("1:59"))).astype(int)
    values_df['Q4_0:40-0:59'] = ((values_df['pff_QUARTER'] == 4) & values_df['pff_CLOCK'].between(timeToSecs("0:40"), timeToSecs("0:59"))).astype(int)
    values_df['Q4_0:20-0:39'] = ((values_df['pff_QUARTER'] == 4) & values_df['pff_CLOCK'].between(timeToSecs("0:20"), timeToSecs("0:39"))).astype(int)
    values_df['Q4_0:08-0:19'] = ((values_df['pff_QUARTER'] == 4) & values_df['pff_CLOCK'].between(timeToSecs("0:08"), timeToSecs("0:19"))).astype(int)
    values_df['Q4_0:00-0:07'] = ((values_df['pff_QUARTER'] == 4) & values_df['pff_CLOCK'].between(timeToSecs("0:00"), timeToSecs("0:07"))).astype(int)
    return values_df

def getCertainty(model, values_df):
    tree_predictions = [tree.predict(values_df)[0] for tree in model.estimators_]
    from collections import Counter
    counts = Counter(tree_predictions)
    total = sum(counts.values())
    percentages = {cls: count / total * 100 for cls, count in counts.items()}
    percent_pass = percentages.get(1, 0)
    percent_run = percentages.get(0, 0)
    return percent_pass if percent_pass > percent_run else percent_run

def predict(values):
    # Load the model
    model = load('random_forest_model.joblib')
    # This will eventually be the values received from the page. For now, we are going to hard code a row for simplicity's sake.
    values_df = createDataframe(values)
    print("Values dataframe: ", values_df)
    pred_df = model.predict(values_df)
    pred = pred_df[0]
    print("Predicted: ", pred)
    certainty = round(getCertainty(model, values_df))
    return [int(pred), certainty]

# Run Server
app = Flask(__name__)
CORS(app)

@app.route('/send_strings', methods=['POST'])
def receive_strings():
    data = request.json  # Expecting JSON data from the client
    values = data.get("strings", [])  # Get the list of strings
    
    pred = predict(values)
    
    return jsonify({"message": "Prediction made successfully!", "received": pred})


if __name__ == '__main__':
    app.run(port=5000, debug=True)  # Runs on localhost:5000


