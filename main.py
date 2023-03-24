import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn import preprocessing
import pickle
app_test=pd.read_pickle("good_app_test.pkl")

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    SK_ID_CURR = int(request.form['SK_ID_CURR'])
    features = app_test.loc[app_test['SK_ID_CURR'] == SK_ID_CURR]
    features = features.drop(['SK_ID_CURR'], axis=1)
    final_features = np.array(features)
    
    pred_proba = model.predict_proba(features)
    output = pred_proba[0][1]
    

    return render_template('index.html', prediction_text='Le taux de probabilité que ce client ne rembourse pas est de {}'.format(output))



if __name__ == "__main__":
    app.run(port=3000)