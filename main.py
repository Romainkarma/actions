import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn import preprocessing
import pickle

app_test=pd.read_pickle("good_app_test.pkl")
app_train=pd.read_pickle("good_app_train_light.pkl")

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return "Home page"

@app.route('/predict', methods=['GET'])
def predict():
    SK_ID_CURR = int(request.args.get('SK_ID_CURR'))
    features = app_test.loc[app_test['SK_ID_CURR'] == SK_ID_CURR]
    features = features.drop(['SK_ID_CURR'], axis=1)
    final_features = np.array(features)

    pred_proba = model.predict_proba(features)
    output = pred_proba[0][1]
    if output > 0.2:
        reponse = 1
    else:
        reponse = 0
    
    # Get additional information
    days_birth = features.iloc[0]['DAYS_BIRTH']
    ext_source_2 = features.iloc[0]['EXT_SOURCE_2']
    ext_source_3 = features.iloc[0]['EXT_SOURCE_3']
    
    days_birth_list = list(app_train['DAYS_BIRTH'])
    ext_source_2_list = list(app_train['EXT_SOURCE_2'])
    ext_source_3_list = list(app_train['EXT_SOURCE_3'])
    target_train = list(app_train['TARGET'])

    response = {
        'SK_ID_CURR': SK_ID_CURR,
        'prediction': output,
        'reponse': reponse,
        'DAYS_BIRTH': days_birth,
        'EXT_SOURCE_2': ext_source_2,
        'EXT_SOURCE_3': ext_source_3,
        'DAYS_BIRTH_LIST': days_birth_list,
        'EXT_SOURCE_2_LIST': ext_source_2_list,
        'EXT_SOURCE_3_LIST': ext_source_3_list,
        'TARGET_TRAIN': target_train
    }


    return jsonify(response)

if __name__ == "__main__":
    app.run()
