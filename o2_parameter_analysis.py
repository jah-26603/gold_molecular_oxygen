# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 10:09:24 2026

@author: JDawg
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import functions

from sklearn.model_selection import train_test_split

import xgboost as xgb
import shap

from functions.load_data import load_o2wNe
from datetime import timedelta

gold_on2 = False
icon_on2 = False
guvi_on2 = False
nam_sam_indices = True
cosmic_ne = False

cols = ['o2_lon', 'o2_lat', 'doy', 'o2_lt', 'f107', 'ap']

o2_df = functions.load_gold_o2('data\o2den')
o2_df['year'] = o2_df.o2_datetime.dt.year

#%% load data
df = o2_df.copy(deep = True)
if gold_on2:
    #load
    gold_on2_df = functions.load_gold_on2(r'D:\on2\gold', max_year = 2025)
    
    #merge onto o2 dataframe
    df['ut_bins'] = pd.cut(o2_df['o2_ut'], bins=np.arange(0,23, 1.5), labels=False, include_lowest=True)
    gold_on2_df['ut_bins'] = pd.cut(gold_on2_df['on2_ut'], bins=np.arange(0,23, 1.5), labels=False, include_lowest=True)
    gdf = pd.merge(df, gold_on2_df, how = 'left', on = ['date', 'ut_bins'])
    mask = ((np.abs(gdf.o2_lat - gdf.on2_lat) <= 7.5) &
                (np.abs(gdf.o2_lon - gdf.on2_lon) <= 30) &
                (np.abs(gdf.on2_lt - gdf.o2_lt) <= 3) &
                (np.abs(gdf.o2_datetime - gdf.on2_datetime) <= timedelta(minutes=40)))
    
    
    #remove meausurements outside of the criteria
    cols_to_nan = ['on2', 'on2_lon', 'on2_lat', 'on2_ut', 'on2_lt']
    gdf.loc[~mask, cols_to_nan] = np.nan
        
    #on2 may need to be averaged out?
    mm = gdf.groupby(gdf.o2)[cols_to_nan].agg('mean')
    df = pd.merge(df,mm, on = 'o2')
    cols.extend(cols_to_nan)

if icon_on2:
    #load
    icon_df = functions.load_icon_on2(r"D:\on2\icon\l2-4_fuv_day\numpy_icondiskon2.npy.npz")
    icon_df['on2_ut'] = (icon_df['icondisktime'].dt.hour + icon_df['icondisktime'].dt.minute / 60 + icon_df['icondisktime'].dt.second / 3600)
    
    
    #merge onto o2 dataframe
    df['ut_bins'] = pd.cut(df['o2_ut'], bins=np.arange(0,23, 1.5), labels=False, include_lowest=True)
    icon_df['ut_bins'] = pd.cut(icon_df['on2_ut'], bins=np.arange(0,23, 1.5), labels=False, include_lowest=True)
    idf = pd.merge(df, icon_df, how = 'left', on = ['date', 'ut_bins'])
    mask = ((np.abs(idf.icondisklat - idf.o2_lat) <= 7.5) &
                (np.abs(idf.icondisklon - idf.o2_lon) <= 15) &
                (np.abs(idf.icondisklt - idf.o2_lt) <= 1) &
                (np.abs(idf.on2_ut - idf.o2_ut) <= 2/3)
                )
    
    
    #remove meausurements outside of the criteria
    cols_to_nan = ['icondiskon2', 'icondisklon', 'icondisklat', 'on2_ut', 'icondisklt']
    idf.loc[~mask, cols_to_nan] = np.nan
        
    #on2 may need to be averaged out?
    mm = idf.groupby(idf.o2)[cols_to_nan].agg('mean')
    df = pd.merge(df,mm, on = 'o2')
    df['on2'] = df.icondiskon2
    cols.extend(cols_to_nan)



    
if nam_sam_indices:
    #load, add 7 day time lag, and merge
    nam_sam = functions.nam_sam_idx(fp = r"data\MERRA2_dTdy_Ubar_SSW_NAM+SAM_19800101-20240331_3d.nc")
    nam_sam['date'] = pd.to_datetime(dict(year=nam_sam.year, month=nam_sam.month, day=nam_sam.day)).astype(object)
    nam_sam['date'] = nam_sam.date + pd.Timedelta( days = 7)
    df = pd.merge(df, nam_sam, how = 'left', on = ['date'])
    
    cols.extend(['nam','sam'])

if cosmic_ne:
    #load
    df = load_o2wNe(df, r'D:\cosmic2_gis')
    df = df.reset_index(drop=True)
    
    #data clean
    mask = ((df.o2_sza > 75)) #only want daytime plasma densities
    pc = .995
    df.loc[mask,'Ne_den'] = np.nan
    df.loc[df.Ne_den > df.Ne_den.quantile(pc), 'Ne_den'] = np.nan #after this it goes up wildly...
    df.loc[df.Ne_den < df.Ne_den.quantile(1 - pc), 'Ne_den'] = np.nan #after this it goes up wildly... quantile is .00125
    
    cols.append('Ne_den')






#%% train models and gather shapley values

epochs =30

# cols = ['o2_lon', 'o2_lat', 'doy', 'o2_lt', 'f107', 'ap']
# cols.extend(cols_to_nan)
# cols.append('on2')
# cols.extend(['nam','sam'])
# cols.append('Ne_den')
train_loss, test_loss, r_corr = [], [], []
# df = df.dropna(axis = 0) 
for i in range(epochs):

    X = df[cols].astype(float)
    y = df.o2
    X_train, X_test, y_train, y_test = train_test_split(X.astype('float'), y, test_size=0.30, random_state = i)
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(X.columns))
    dvalid = xgb.DMatrix(X_test, label=y_test, feature_names=list(X.columns))
    
    param = {
        'subsample': 1,
        "max_depth": 4,
        "objective": "reg:absoluteerror",  #
        'reg_lambda' :7e9, #needs to be on order of magnitude as y
        'eval_metric' : 'mape',
        "device": "cuda"
    }
    evals = [(dtrain, "train"), (dvalid, "valid")]
    evals_result = {}
    # Define model
    model = xgb.train(
        params=param,
        dtrain=dtrain,
        num_boost_round=300,
        evals=evals,
        early_stopping_rounds=20,
        evals_result=evals_result,
        verbose_eval=50
    )
    
    train_loss.append(evals_result['train']['mape'][-1])
    test_loss.append(evals_result['valid']['mape'][-1])
    
    dtotal = xgb.DMatrix(X, label=y[X.index], feature_names=list(X.columns))
    shap_values = model.predict(dvalid, pred_contribs=True)
    shap_df = pd.DataFrame(shap_values, columns=list(X.columns) + ["bias"])
    

    explainer = shap.TreeExplainer(model)
    shap_values_full = explainer.shap_values(X)/y.mean()
    
    

    # # ###PLOTTING
    # # #PLOT EVERY SHAPLEY VALUES
    # features = list(X.columns)
    # n_features = len(features)
    # n_rows = 2
    # n_cols = int(np.ceil(n_features / n_rows))
    
    # fig, axes = plt.subplots(n_rows, n_cols, figsize=(5* n_cols, 3 * n_rows), squeeze=False)
    # axes = axes.flatten()
    # #all axes
    # for ki, f in enumerate(np.array(features)[(np.abs(shap_values_full).mean(axis = 0)).argsort()[::-1]]):
    #     shap.dependence_plot(f, shap_values_full * 100, X, ax=axes[ki], show=False, interaction_index=f)
    #     axes[ki].set_ylabel(f'% deviation for {f}')
        
    # plt.suptitle(f"GOLD O₂ Shapley Value Interactions Seed {i+1}", fontsize=16)
    # plt.tight_layout()
    # plt.show()
    
    

    # # #BEESWARM PLOT
    # plt.figure(figsize=(10,10))   # ← square figure
    # shap.summary_plot(shap_values_full* 100, X, plot_type="dot", show=False)
    # plt.title(f"GOLD O₂ Seed {i+1}", fontsize=16)
    # plt.tight_layout()
    # plt.xlabel('Shap value (impact on model output) % deviation from mean')
    # plt.xlim(-1.5e8*100/ y.mean() ,1.5e8 *100/ y.mean())
    # plt.show()
    var = 'nam'

    mask = np.ones_like(X)[:,0].astype(bool)
    mask = ~np.isnan(X[var])
    mask = np.abs(X.nam)> 1
    plt.figure()
    shap.dependence_plot(var, shap_values_full[mask] * 100, X[mask], show = False)
    plt.title(np.corrcoef(shap_values_full[:,-2][mask], X[var][mask])[1,0])
    plt.show()
    


    r_corr.append(np.corrcoef(shap_values_full[:,-2][mask], X[var][mask])[1,0])



import seaborn as sns
# plt.figure()
# sns.histplot(train_loss, label = 'train loss')
# sns.histplot(test_loss, label = 'val loss')
# plt.vlines(np.mean(test_loss), 0 , 16, color = 'red', label = f' pearson_r = {round(np.mean(test_loss),3)}')
# plt.xlim(.1, .15)
# plt.title('climatology (nam/sam sampled)')
# plt.legend()
# plt.show()



plt.figure()
sns.histplot(r_corr)
plt.xlim(-1,1)
plt.title('Ne_den (160-200km)')
plt.show()