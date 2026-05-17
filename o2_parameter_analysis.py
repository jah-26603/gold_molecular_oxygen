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

gold_on2 = True
nam_sam_indices = True
cosmic_ne = True

icon_on2 = False
guvi_on2 = False


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
    cols_to_nan = ['on2'
                   #, 'on2_lon', 'on2_lat', 'on2_ut', 'on2_lt'
                   ]
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
    nam_sam = nam_sam[['nam', 'sam', 'date']]
    df = pd.merge(df, nam_sam, how = 'left', on = ['date'])
    
    cols.extend(['nam','sam'])

if cosmic_ne:
    #load
    df = load_o2wNe(df, r'D:\cosmic2_gis')
    df = df.reset_index(drop=True)    
    
    #data clean
    mask = ((df.o2_sza > 75)) #only want daytime plasma densities
    pc = .995
    df.loc[mask,'ne_160_200km'] = np.nan
    df.loc[mask,'TEC'] = np.nan
    df.loc[df.ne_160_200km > df.ne_160_200km.quantile(pc), 'ne_160_200km'] = np.nan #after this it goes up wildly...
    df.loc[df.ne_160_200km < df.ne_160_200km.quantile(1 - pc), 'ne_160_200km'] = np.nan #after this it goes up wildly... quantile is .00125
    df.loc[df.TEC > df.TEC.quantile(pc), 'TEC'] = np.nan #after this it goes up wildly...
    df.loc[df.TEC < df.TEC.quantile(1 - pc), 'TEC'] = np.nan #after this it goes up wildly... quantile is .00125
    
    cols.append(['TEC', 'Ne_den'])






#%%
# train models and gather shapley values

epochs = 25
train_loss, test_loss, r_corr = [], [], []

it_params = {
            'on2': {'var': ['on2']},
             'ne_160_200km': {'var': ['ne_160_200km']}, 
             'TEC': {'var': ['TEC']},
             'polar_vortex': {'var': ['nam', 'sam']},
             }

for k in it_params.keys():
    
    var = it_params[k]['var']
    cols = ['o2_lon', 'o2_lat', 'doy', 'o2_lt', 'f107', 'ap']
    cols.extend(var)

    it_params[k]['train_loss'] = []
    it_params[k]['test_loss'] = []
    it_params[k]['r_corr'] = []
    
        
    for i in range(epochs):
        
        X = df[cols].astype(float)
        y = df.o2
        X_train, X_test, y_train, y_test = train_test_split(X.astype('float'), y, test_size=0.30, random_state = i)
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(X.columns))
        dvalid = xgb.DMatrix(X_test, label=y_test, feature_names=list(X.columns))
        
        reg_lambda = 7e9
        if (k == 'ne_160_200km') or (k == 'TEC'):
            reg_lambda = 7e9 #higher regularization for plasma data
            
        param = {
            'subsample': 1,
            "max_depth": 4,
            "objective": "reg:absoluteerror",  #
            'reg_lambda' :reg_lambda, 
            # 'reg_lambda' :0, 
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
    
        it_params[k]['train_loss'].append(evals_result['train']['mape'][-1])
        it_params[k]['test_loss'].append(evals_result['valid']['mape'][-1])
        
        dtotal = xgb.DMatrix(X, label=y[X.index], feature_names=list(X.columns))
        shap_values = model.predict(dvalid, pred_contribs=True)
        shap_df = pd.DataFrame(shap_values, columns=list(X.columns) + ["bias"])
        
    
        explainer = shap.TreeExplainer(model)
        shap_values_full = explainer.shap_values(X)#/y.mean()
        
        
    
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
        
        if k == 'polar_vortex':
            xam = it_params[k]['var']
            for j in range(len(xam)):
                mask = np.abs(X[xam[j]])> 1
                var = xam[j]
                
                idx = len(cols) - len(xam) + j
                # plt.figure()
                # shap.dependence_plot(var, shap_values_full[mask] * 100, X[mask], show = False)
                # plt.title(np.corrcoef(shap_values_full[:,idx][mask], X[var][mask])[1,0])
                # plt.show()
                it_params[k]['r_corr'].append(np.corrcoef(shap_values_full[:,idx][mask], X[var][mask])[1,0])
        else:
            
            if isinstance(var, list):
                var = var[0]
            
            
            mask = np.ones_like(X)[:,0].astype(bool)
            mask = ~np.isnan(X[var])

            # plt.figure()
            # shap.dependence_plot(var, shap_values_full[mask] * 100, X[mask], show = False)
            # plt.title(np.corrcoef(shap_values_full[:,-1][mask], X[var][mask])[1,0])
            # plt.show()
            
            it_params[k]['r_corr'].append(np.corrcoef(shap_values_full[:,-1][mask], X[var][mask])[1,0])



nam_corr = it_params['polar_vortex']['r_corr'][::2]
sam_corr = it_params['polar_vortex']['r_corr'][1::2]
on2_corr = it_params['on2']['r_corr']
ned_corr = it_params['ne_160_200km']['r_corr']
tec_corr = it_params['TEC']['r_corr']

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def dist_plotting(arr, feature):
    
    plt.figure()
    sns.histplot(arr)
    plt.title(feature + ' correlations')
    plt.vlines(np.mean(arr), 0 , 16, color = 'red', label = f'mean = {round(np.mean(arr), 3)}')
    plt.legend()
    plt.xlim(-1,1)
    plt.show()
    
dist_plotting(ned_corr, 'Ne_den')
dist_plotting(tec_corr, 'TEC')
dist_plotting(nam_corr, 'NAM')
dist_plotting(sam_corr, 'SAM')
dist_plotting(on2_corr, 'O/N2')

#%%
corr_df = pd.DataFrame({
    'Correlation': (
        list(on2_corr) +
        # list(ned_corr) +
        list(tec_corr) +
        list(nam_corr) +
        list(sam_corr) 
    ),
    'Variable': (

        ['O/N2'] * len(on2_corr) +
        # ['Ne Density'] * len(ned_corr) +
        ['TEC'] * len(tec_corr)+
        ['NAM'] * len(nam_corr) +
        ['SAM'] * len(sam_corr) 
    )
})
g = sns.FacetGrid(
    corr_df,
    row='Variable',
    hue='Variable',
    aspect=5,
    height=1.2,
    sharey=False

)

g.map(
    sns.kdeplot,
    'Correlation',
    fill=True,
    bw_adjust=.65
)

g.map(plt.axvline, x=0, linestyle='--')

g.set_titles("{row_name}")
g.set_xlabels("Correlation")
g.set_ylabels("Density")
plt.xlim(-1,1)
plt.tight_layout()
plt.show()