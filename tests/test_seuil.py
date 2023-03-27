import pickle
import pandas as pd
import numpy as np
  
app_test=pd.read_pickle("good_app_test.pkl")
model = pickle.load(open('model.pkl', 'rb'))
sample = app_test.sample(frac=0.01)
results=[]
seuil=0.1

SK_ID_CURR=sample["SK_ID_CURR"]
for value in SK_ID_CURR:
    features = sample.loc[sample['SK_ID_CURR'] == value]
    features = features.drop(['SK_ID_CURR'], axis=1)
    final_features = np.array(features)

    pred_proba = model.predict_proba(features)
    output = pred_proba[0][1]
    results.append(output)

def test_seuil():
    
    average = sum(results) / len(results)
    assert average <= seuil, f"La moyenne {average} dépasse la valeur seuil de 0.2"
    