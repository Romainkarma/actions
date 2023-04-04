import pickle
import pandas as pd
import numpy as np
import sklearn.metrics as metrics

sample_n=pd.read_pickle("sample_n.pkl")
seuil_score = 5000

def coastScorer(y_val, y_pred):
    cm = metrics.confusion_matrix(y_val, y_pred).ravel().tolist()
    cost = 20*cm[2]- 0*cm[0] + 2*cm[1] - 2*cm[3]
    return cost

model = pickle.load(open('model.pkl','rb'))

X_val=sample_n.drop(columns=["TARGET"])
y_val=sample_n["TARGET"]

y_pred = model.predict_proba(X_val)
y_pred_ok = y_pred[:, 1]

y_pred_prob = (y_pred_ok > 0.2).astype('int')
coastscore = coastScorer(y_val, y_pred_prob)

def test_seuil():
    
    assert coastscore > seuil_score, f"Le score métier {coastscore} dépasse le score pour la valeur seuil de 0.2"
    