# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 11:08:09 2026

@author: dogbl
"""

from scipy.io import readsav
import pandas as pd
import numpy as np
from pymsis.utils import get_f107_ap
import matplotlib.pyplot as plt
import xgboost as xgb
import shap

mission = 'SOLSTICE'
mission = 'SUSIM'

if mission == 'SUSIM':
    data = readsav(r"C:\Users\JDawg\Downloads\SUSIM_O2.sav")
    df_data = np.stack([data['year'], data['doy'], data['ut'], 
               data['ltime'], data['lon'],
               data['o2_density'][:,16], data['lat']])

if mission == 'SOLSTICE':
    data = readsav(r"C:\Users\JDawg\Downloads\SOLSTICE_O2.sav")
    df_data = np.stack([data['year'], data['doy'], data['ut'], 
               data['ltime'], data['lon'],
               data['o2_density'][:,13], data['lat']])


cols = ['year', 'doy', 'ut', 'lt', 'lon', 'o2', 'lat']
df = pd.DataFrame(df_data.T, columns = cols)

df['date'] = pd.to_datetime(df['year'], format='%Y') + pd.to_timedelta(df['doy'] - 1, unit='D')
df['datetime'] = df['date'] + pd.to_timedelta(df['ut'], unit='h')

f107, _, ap = get_f107_ap(df.datetime)
df['f107'] = f107
df['ap'] = ap[:,1]

df = df[df.o2 != -99]
i = 0

y = df['o2']

X = df[['lon','lat','doy','lt','f107','ap']].astype(float)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.50, random_state=42
)

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(X.columns))
dvalid = xgb.DMatrix(X_test, label=y_test, feature_names=list(X.columns))

reg_lam = 2.95e8
num_est = 10

if mission == 'SOLSTICE':
    reg_lam = 3.95e8
    num_est = 45
param = {
    "subsample": 1,
    "max_depth": 4,
    "objective": "reg:absoluteerror",
    "reg_lambda":reg_lam,
    "eval_metric": "mape",
    "device": "cuda",
}

evals = [(dtrain, "train"), (dvalid, "valid")]
evals_result = {}

model = xgb.train(
    params=param,
    dtrain=dtrain,
    num_boost_round=num_est,
    evals=evals,
    early_stopping_rounds=20,
    evals_result=evals_result,
    verbose_eval=10
)

# SHAP
shap_x = X_test
explainer = shap.TreeExplainer(model)
shap_values_full = explainer.shap_values(shap_x)
mask = shap_x.lat < 0
for i, feature in enumerate(X.columns):

    # if feature != 'f107':
    #     continue
    plt.figure()

    shap.dependence_plot(
        feature,
        shap_values_full[mask] / y.mean() * 100,
        shap_x[mask],
        # interaction_index='lat',
        alpha=0.3,
        show=False
    )

    plt.ylabel('Percent deviation (%)')
    plt.title(f"{mission} O2: Functional Behavior of {feature}")
    # plt.xlim(60, 330)
    plt.show()
    
    
    
    
    
  


