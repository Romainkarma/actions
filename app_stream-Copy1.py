import numpy as np
import pandas as pd
import streamlit as st
from sklearn import preprocessing
import pickle

app_test = pd.read_pickle("good_app_test.pkl")
model = pickle.load(open('model.pkl', 'rb'))

def predict(SK_ID_CURR):
    features = app_test.loc[app_test['SK_ID_CURR'] == SK_ID_CURR]
    features = features.drop(['SK_ID_CURR'], axis=1)
    final_features = np.array(features)
    pred_proba = model.predict_proba(features)
    output = pred_proba[0][1]
    return output

def main():
    st.title('Prediction App')
    SK_ID_CURR = st.text_input('Enter SK_ID_CURR:', value='')
    if SK_ID_CURR:
        try:
            SK_ID_CURR = int(SK_ID_CURR)
            output = predict(SK_ID_CURR)
            st.success('Le taux de probabilité que ce client ne rembourse pas est de {}'.format(output))
        except:
            st.error('Invalid input.')

if __name__ == "__main__":
    main()
