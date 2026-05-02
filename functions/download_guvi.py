# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 15:05:13 2025

@author: JDawg
"""

import requests
import os
from tqdm import tqdm

base_url = "https://guvitimed.jhuapl.edu/"

years = range(2024, 2024 + 1)   # adjust range as needed
days = range(1, 366)

outdir = r"E:\guvi_on2_data"
os.makedirs(outdir, exist_ok=True)

for y in tqdm(years, desc="Years"):
    os.makedirs(os.path.join(outdir,str(y)), exist_ok=True)
    for d in tqdm(days, desc=f"Days in {y}", leave=False):
        fname = f"data/level3/level3_on2_spect/netcdf/{y}/{str(d).zfill(3)}/timed_guvi_l3-on2_{y}{str(d).zfill(3)}_Av0100r000.nc"
        url = f"{base_url}{fname}"
        outpath = os.path.join(os.path.join(outdir,str(y)), os.path.basename(fname))

        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                with open(outpath, "wb") as f:
                    f.write(r.content)
                print("Downloaded:", fname)
            else:
                print("Not found:", url)
        except Exception as e:
            print("Failed:", fname, e)