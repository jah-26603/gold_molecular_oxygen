# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 16:45:19 2025

@author: JDawg
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import statsmodels.api as sm

def gold_trad_outliers(df):
    df['dist'] = ((df['o2_lat'] - df['on2_lat'])**2 + (df['o2_lon'] - df['on2_lon'])**2)**.5 # this should be great circle but its close enough
    pdf = (df.sort_values('dist').groupby('o2_datetime').head(10).groupby('o2_datetime')[['on2','on2_lat', 'on2_lt']].median()) #selects 10 nearest pixels
    pdf = pdf.rename(columns = {'on2' : 'med_on2', 'on2_lat': 'med_on2_lat', 'on2_lt': 'med_on2_lt'})
    df = pd.merge(df, pdf, on = 'o2_datetime', how = 'right')
    
    
    o_df = df.copy().reset_index(drop = True)
    
    # # Define bins
    doy_bins = np.linspace(1, 366, 26)  # 20 bins
    f10_bins = np.linspace(1, 101, 4)  # 10 bins
    lt_list =  np.linspace(0, 24, num= 24, endpoint=False)
    lat_list = np.linspace(-65, 50, num=13, endpoint=False)

    
    
    # # Bin the features
    # # on2 has a lot higher resolution, so we can bin vs finer lat and lt resolution
    o_df['on2_lat_bin'] = pd.cut(o_df['med_on2_lat'], bins=lat_list, labels=False, include_lowest=True)
    o_df['on2_lt_bin'] = pd.cut(o_df['med_on2_lt'], bins=lt_list, labels=False, include_lowest=True)
    o_df['doy_bin'] = pd.cut(o_df['doy'], bins=doy_bins, labels=False, include_lowest=True)
    
    o_df['o2_lat_bin'] = pd.cut(o_df['o2_lat'], bins=lat_list, labels=False, include_lowest=True)
    o_df['o2_lt_bin'] = pd.cut(o_df['o2_lt'], bins=lt_list, labels=False, include_lowest=True)

    threshold = 50  # minimum datapoints required

    on2_df = (
        o_df.groupby(['on2_lat_bin', 'on2_lt_bin', 'doy_bin'])
        .agg(
            on2_model_median=('med_on2', 'median'),
            on2_model_std=('med_on2', 'std'),
            count=('med_on2', 'count')
        )
        .reset_index()
    )
    

    
    
    o2_df = (
        o_df.groupby(['o2_lat_bin', 'o2_lt_bin', 'doy_bin'])
        .agg(
            o2_model_median=('o2', 'median'),
            o2_model_std=('o2', 'std'),
            count=('o2', 'count')
        )
        .reset_index()
    )
    # mask std where count < threshold
    on2_df.loc[on2_df['count'] < threshold, 'on2_model_std'] = np.nan
    o2_df.loc[o2_df['count'] < threshold, 'o2_model_std'] = np.nan

    o_df = o_df.merge(on2_df, how = 'left', on = ['on2_lat_bin', 'on2_lt_bin', 'doy_bin'])
    o_df = o_df.merge(o2_df, how = 'left', on = ['o2_lat_bin', 'o2_lt_bin', 'doy_bin'])
    
    o_df['on2_residual'] = o_df.med_on2 - o_df.on2_model_median
    o_df['o2_residual'] = o_df.o2 - o_df.o2_model_median
    
    o_df['on2_residual_pct'] = 100 * (o_df.med_on2 - o_df.on2_model_median) / o_df.on2_model_median
    o_df['o2_residual_pct']  = 100 * (o_df.o2      - o_df.o2_model_median)  / o_df.o2_model_median
    outlier_mask = ((np.abs(o_df.on2_residual) > 2*o_df.on2_model_std) & (np.abs(o_df.o2_residual) > 2*o_df.o2_model_std))
    
    
    
    X = o_df.o2_residual_pct[outlier_mask]
    y = o_df.on2_residual_pct[outlier_mask].convert_dtypes()

    # Add a constant column for intercept
    X_with_const = sm.add_constant(X)

    # Fit the OLS model
    model = sm.OLS(y, X_with_const)
    results = model.fit()

    # Generate x values for plotting the regression line
    x_plot = np.linspace(X.min(), X.max(), 50)
    x_plot_with_const = sm.add_constant(x_plot)
    y_pred = results.predict(x_plot_with_const)
    # Compute Pearson r
    r, p_value = pearsonr(X, y)
    print(f"Pearson r: {r:.3f}, p-value: {p_value:.3e}")
    
    plt.figure()
    plt.scatter(X,y)
    plt.plot(x_plot, y_pred)
    plt.text(
    0.05, 0.95,
    f"r = {r:.3f}",
    transform=plt.gca().transAxes,   # relative coords (0–1)
    ha="left", va="top",
    fontsize=12, color="blue"
    )

    plt.xlabel("O2 residual (% from median)")
    plt.ylabel("O/N2 residual (% from median)")
    plt.title("Relative residuals (percent scale)")
    plt.axhline(0, color='k', linestyle='--', alpha=0.6)
    plt.axvline(0, color='k', linestyle='--', alpha=0.6)
    plt.show()



    
    return o_df