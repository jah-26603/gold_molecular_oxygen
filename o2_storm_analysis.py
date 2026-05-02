# -*- coding: utf-8 -*-
"""
Created on Fri May  1 11:09:16 2026

@author: JDawg
"""

import os
import glob
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap
import functions
from pymsis.utils import get_f107_ap
import seaborn as sns
import spaceweather as sw
from itertools import product

intial_run = False #Only do this if the icon-tiegcm data is not downloaded
if intial_run:
    functions.download_icon_tiegcm(fp = r'D:\icon_tiegcm')
    functions.icon_tiegcm_o2_data(fp = r'D:\icon_tiegcm', out = 'icon_tiegcm_climatology.csv')
    
#GOLD Data Frame
o2_df = functions.load_gold_o2('data\o2den')
o2_df['year'] = o2_df.o2_datetime.dt.year


#ICON-TIEGCM data Frame
idf = pd.read_csv("icon_tiegcm_climatology.csv", index_col = 0)
idf['datetime'] = pd.to_datetime(idf.year.astype(str) + idf.doy.astype(str).str.zfill(3) + idf.ut_hour.astype(str).str.zfill(2), format = '%Y%j%H',errors='coerce')
idf = idf.dropna(subset=['datetime'])


#%% #Copies of the original dataframes, makes it useful for data manipulation.

#GOLD DATAFRAME COPY --> Switch Ap to Kp
go_df = o2_df[['o2', 'o2_lat', 'o2_lon', 'o2_lt', 'doy', 'f107',  'o2_datetime']].copy()
kp = sw.ap_kp_3h()
kp['o2_datetime'] = pd.to_datetime(kp.index)
go_df['Kp'] = pd.merge_asof(go_df, kp, on = 'o2_datetime')['Kp']


#ICON-TIEGCM COPY
ic_df = idf[['o2', 'lat', 'lon', 'lt', 'doy', 'f107', 'Kp', 'datetime']].copy() 
ic_df = (
    ic_df[['o2', 'lat', 'lon', 'lt', 'doy', 'f107', 'Kp', 'datetime']]
    .rename(columns={
        'lat': 'o2_lat',
        'lon': 'o2_lon',
        'lt': 'o2_lt',
        'datetime': 'o2_datetime'
    })
    .copy()
)



# This selects a 2 week window of dates centered around dates that are below a dst threshold.
# This will push the distribution of Ap-f107 for ICON closer to that of GOLD.
df = pd.read_csv(r'https://lasp.colorado.edu/space-weather-portal/latis/dap/kyoto_dst_index_service.csv?time,dst&time>=2020-01-01T00:00:00Z&time<=2023-01-01T00:00:00Z', names = ['time', 'dst'], skiprows = 1)
df['time'] = pd.to_datetime(df.time)
kk = df[df.dst < -40]
dates = []
for days in np.arange(-7, 7 + 1):
    new_dates = kk.time.dt.date.unique() + pd.Timedelta(days = days)
    dates.append(new_dates)
    
dates = np.hstack(dates)
dates = list(set(dates))
dates.sort()
ic_df = ic_df[ic_df['o2_datetime'].dt.date.isin(dates)]


#%% Train Ensemble of XGB models

def train_xgb_models(cols, target, df, icon = False):
    models = []
    num_models = 10 #number of models we combine for prediction
    if icon:
        df = df.sample(frac = 1)
        
    for i in range(num_models):
        X = df[cols]
        X['sin_lt'] = np.sin(2*np.pi * X['o2_lt'] / 24)
        X['cos_lt'] = np.cos(2*np.pi * X['o2_lt'] / 24)
        X = X.drop(columns = 'o2_lt')
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X.astype('float'), y, test_size=0.30, random_state = i)
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(X.columns))
        dvalid = xgb.DMatrix(X_test, label=y_test, feature_names=list(X.columns))
        param = {
            'subsample': 1,
            "max_depth": 4,
            "objective": "reg:squarederror",  #
            'reg_lambda' :1e0,
            'eval_metric' : 'mae',
            "device": "cuda"
        }
        evals = [(dtrain, "train"), (dvalid, "valid")]
        evals_result = {}
        # Define model
        model = xgb.train(
            params=param,
            dtrain=dtrain,
            num_boost_round=500,
            evals=evals,
            early_stopping_rounds=200,
            evals_result=evals_result,
            verbose_eval=50
        )
        model.set_param({"device": "cuda"})
        models.append(model)
        
        
        
        # dtotal = xgb.DMatrix(X, label=y[X.index], feature_names=list(X.columns))
        # shap_values = model.predict(dtotal, pred_contribs=True)
        # shap_df = pd.DataFrame(shap_values, columns=list(X.columns) + ["bias"])
        
        
        # explainer = shap.TreeExplainer(model)
        # shap_values_full = explainer.shap_values(X)
        # plt.figure(figsize=(10,10))   # ← square figure
        # shap.summary_plot(shap_values_full, X, plot_type="dot", show=False)
        # plt.title("ICON O₂", fontsize=16)
        # plt.tight_layout()
        # plt.show()
        
        # for i,feature in enumerate(X.columns):
            
        #     # if feature == 'sam':
        #     #     mask = ~mask
        #     plt.figure()
        #     shap.dependence_plot(feature, shap_values_full/y.mean() * 100, X, show = False, alpha = .3)
        #     plt.show()
        
    return models
    
def eval_xgb(X, models, lt_only = False):
    
    if not lt_only:
        X['sin_lt'] = np.sin(2*np.pi * X['o2_lt'] / 24)
        X['cos_lt'] = np.cos(2*np.pi * X['o2_lt'] / 24)
        X = X.drop(columns = 'o2_lt')
        pass

    else:        
        X['sin_lt'] = np.sin(2*np.pi * X['o2_lt'] / 24)
        X['cos_lt'] = np.cos(2*np.pi * X['o2_lt'] / 24)
        X = X[['sin_lt', 'cos_lt']]
    X = xgb.DMatrix(X)
    y = []
    for i in range(len(models)):
        y.append(models[i].predict(X))
        
    return np.mean(np.stack(y), axis = 0)


#train models
cols = ['o2_lat', 'o2_lon', 'o2_lt', 'doy', 'f107', 'Kp']
target = 'o2'

#the error should look super high since im not normalizing the data before training.
icon_models = train_xgb_models(cols, target, ic_df.copy(), icon = True)
gold_models = train_xgb_models(cols, target, go_df.copy())


#%% High Activity Maps

lts   = np.arange(0, 24 , 2)
doys  = [360, 270,180,90][::-1]
lats  = np.arange(-60,45 + 1, 5)
lons = [33]
Kp = [1]
f107 = [80]
center_cmap = False

#Feel free to change the strength for Kp and F10.7
for strength in np.linspace(5,6, 1):
    for flux in np.linspace(140,200, 1):
        
        
        grid = list(product(lats, lons, lts, doys, f107, Kp))
        df_eval = pd.DataFrame(grid, columns=["o2_lat", "o2_lon", "o2_lt", "doy", "f107", 'Kp'])
        
        icon_quiet = eval_xgb(df_eval, icon_models)
        gold_quiet = eval_xgb(df_eval, gold_models)

        
        df_eval.loc[:,'Kp']  = strength
        df_eval.loc[:,'f107'] = flux

        icon_storm = eval_xgb(df_eval, icon_models)
        gold_storm = eval_xgb(df_eval, gold_models)
        
        
        mape_icon = (icon_storm - icon_quiet)/ (icon_quiet.mean())
        mape_gold = (gold_storm - gold_quiet)/ (gold_quiet.mean())
        

        df_eval['icon_pct'] = mape_icon * 100
        df_eval['gold_pct'] = mape_gold * 100
        
        icon_vmax = df_eval.icon_pct.quantile(1)
        gold_vmax = df_eval.gold_pct.quantile(1)
        icon_vmin = df_eval.icon_pct.quantile(0)   
        gold_vmin = df_eval.gold_pct.quantile(0)   

        # gold_vmax = max(icon_vmax, gold_vmax)
        
        nrows = len(lons) * len(doys)
        ncols = 2
        panel = 1
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(16, 4*nrows)
        )
        
        axes = np.atleast_2d(axes)
        
        for row, lon in enumerate(lons):
            for j, doy in enumerate(doys):
        
                r = row * len(doys) + j
        
                sub = df_eval[
                    (df_eval['o2_lon'] == lon) &
                    (df_eval['doy'] == doy)
                ].copy()
        
                icon_pivot = sub.pivot(
                    index='o2_lat',
                    columns='o2_lt',
                    values='icon_pct'
                )
        
                gold_pivot = sub.pivot(
                    index='o2_lat',
                    columns='o2_lt',
                    values='gold_pct'
                )
        
                levels = 11
                if center_cmap:
                    icon_levels = np.linspace(-icon_vmax, icon_vmax, levels)
                    gold_levels = np.linspace(-gold_vmax, gold_vmax, levels)
                else:
                    icon_levels = np.linspace(icon_vmin, icon_vmax, levels)
                    gold_levels = np.linspace(gold_vmin, gold_vmax, levels)
                    
                # ICON
                cf_icon = axes[r, 0].contourf(
                    icon_pivot.columns,
                    icon_pivot.index,
                    icon_pivot.values,
                    levels=icon_levels,
                    cmap="RdBu_r"
                )
        
                axes[r, 0].set_title(
                    f'ICON TIEGCM | Lon={lon}, DOY={doy}'
                )
                axes[r, 0].set_xlabel('LT')
                axes[r, 0].set_ylabel('Lat')
        
                # GOLD
                cf_gold = axes[r, 1].contourf(
                    gold_pivot.columns,
                    gold_pivot.index,
                    gold_pivot.values,
                    levels=gold_levels,
                    cmap="RdBu_r"
                )
        
                axes[r, 1].set_title(
                    f'GOLD | Lon={lon}, DOY={doy}'
                )
                axes[r, 1].set_xlabel('LT')
        
        # leave space at bottom
        fig.tight_layout(rect=[0, 0.08, 1, 0.95])
        
        # bottom colorbars
        cbar_icon = fig.colorbar(
            cf_icon,
            ax=axes[:, 0],
            orientation='horizontal',
            fraction=0.05,
            pad=0.08
        )
        
        cbar_icon.set_label('ICON % change from quiet')
        
        cbar_gold = fig.colorbar(
            cf_gold,
            ax=axes[:, 1],
            orientation='horizontal',
            fraction=0.05,
            pad=0.08
        )
        
        cbar_gold.set_label('GOLD % change from quiet')
        
        fig.suptitle(
            f'Kp = {strength} F10.7 = {flux}',
            fontsize=30
        )
        
        plt.show()

