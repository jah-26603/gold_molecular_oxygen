# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 20:02:04 2026

@author: JDawg
"""

import netCDF4 as nc
import numpy as np
import pandas as pd
import glob
import os
from tqdm import tqdm 

alt = np.linspace(0, 1000, 51)
lat = np.linspace(-90, 90, num=73)
lon = np.linspace(0, 360, num=72, endpoint=False)

data = []
for f in tqdm(glob.glob(os.path.join(r'E:\on2\cosmic2_gis', '*.nc'))):
    
    # if '2022' in f or '2023' in f or '2024' in f or '2025' in f:
    #     continue
    ds = nc.Dataset(f,'r')
    Ne = ds.variables['Ne'][:].data[:,10:-10] #1e5 cm^-3 
    Ne = np.trapz(Ne[5:27]*1e5*1e6,dx = 20*1e3, axis = 0)/1e16 #convert to SI and then normalized TEC

    data.append(Ne.flatten())
    
data = np.hstack(data)    
#%%

x = data.copy()
x = x [x < np.percentile(x, 99.5)]
quartiles = np.quantile(x, [0.025, 0.05, 0.075])
print(quartiles)

quartiles = np.quantile(x, [1-0.025, 1-0.05, 1-0.075])
print(quartiles)



# import numpy as np
# from scipy.stats import skew
#99.0 is good enough

# x = data.copy()
# x = x[np.isfinite(x)]

# for q in [99.9, 99.5, 99, 98, 97, 95]:
#     hi = np.percentile(x, q)
#     xt = x[x <= hi]
#     print(f"q={q:4.1f}%, skew={skew(xt):8.2f}, N={len(xt)}")
