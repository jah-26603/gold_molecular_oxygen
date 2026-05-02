# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 12:36:22 2026

@author: JDawg
"""

import requests
import os
import glob
from tqdm import tqdm
import shutil
import zipfile
from concurrent.futures import ProcessPoolExecutor
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed





def unzip_one(z):
    fp = os.getcwd()

    try:
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(z) as zip_ref:
                zip_ref.extractall(tmp)

            for root, _, files in os.walk(tmp):
                for f in files:
                    src = os.path.join(root, f)
                    dst = os.path.join(fp, f)
                    #  skip if exists
                    if os.path.exists(dst):
                        continue
                        # os.remove(dst)

                    shutil.move(src, dst)

        #this will delete the zip
        os.remove(z)
        return f"ok: {z}"

    except Exception as e:
        return f"skip: {z} ({e})"
    
    

def download_icon_tiegcm(fp = r'D:\icon_tiegcm'):
    '''downloads the zipped ICON-TIEGCM runs and unpacks them. This function
    will take about 24 hours to run'''
    
    os.makedirs(fp, exist_ok=True)
    os.chdir(fp)
    session = requests.Session()
    
    def download_file(url):
        local_filename = url.split('/')[-1]
        if os.path.exists(local_filename):
            return f"skip {local_filename}"

        try:
            with session.get(url, stream=True, timeout=30) as r:
                if r.status_code != 200:
                    return f"missing {local_filename}"

                with open(local_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024*1024):  # 1 MB chunks
                        if chunk:
                            f.write(chunk)

            return f"done {local_filename}"

        except Exception as e:
            return f"error {local_filename}: {e}"
    
    urls = []
    for year in range(2019, 2023):
        for month in range(1, 13):
            for day in range(1, 32):
                urls.append(
                    f'https://spdf.gsfc.nasa.gov/pub/data/icon/l4/tiegcm/{year}/'
                    f'icon_l4-3_tiegcm_{year}-{month:02d}-{day:02d}_v02r000.zip'
                )
    
    # Tune this (start with 4-8)
    MAX_WORKERS = 8
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(download_file, url) for url in urls]
        for f in as_completed(futures):
            print(f.result())
            
            
    
    zip_list = glob.glob("*.zip")
    downloaded_list = glob.glob('*.nc')
    with ProcessPoolExecutor(max_workers=8) as ex:
        results = list(tqdm(ex.map(unzip_one, zip_list), total=len(zip_list)))
    
    for r in results:
        if r.startswith("skip"):
            print(r)



import numpy as np
import pandas as pd
import netCDF4 as nc
from itertools import product


def icon_tiegcm_o2_data(fp = r'D:\icon_tiegcm', out = 'icon_tiegcm_climatology.csv'):
    '''outputs a csv of O2 data from the ICON TIEGCM runs with associated 
    climatological inputs'''
    data = []
    for f in tqdm(glob.glob(os.path.join(fp, '*.nc'))):
        
        try:
            ds = nc.Dataset(f, 'r')

            ut = ds.variables['mtime'][:]
            year = ds.variables['year'][:][0]
            doy = ut[:,0]
            hour = ut[:,1]
        
            lat = ds.variables['lat'][:]
            lon = ds.variables['lon'][:]
            lev = ds.variables['lev'][:]
        
            lt = (hour[None,:] + lon[:,None]//15)%24
        
        
            #1st dimension is time -> hours
            #2nd dimension is midpoint pressure coordinate
            #3rd & 4th dimensions are lat and lon
        
            Z = ds.variables['ZG'][:].data #height
            mmr_O2 = ds.variables['O2'][:].data #mmr
            mmr_N2 = ds.variables['N2'][:].data
            mmr_O1 = ds.variables['O1'][:].data
            mmr_NO = ds.variables['NO'][:].data
            mmr_N4S = ds.variables['N4S'][:].data
            rho_tot = ds.variables['DEN'][:].data* 1e3 #kg/m3 SI
            f107 = ds.variables['f107d'][:].data
            kp = ds.variables['Kp'][:].data
            avgo = 6.02214076e23  # mol^-1
        
            # molecular masses (kg)
            mo2  = 32e-3 / avgo
            mn2  = 28e-3 / avgo
            mo1  = 16e-3 / avgo
            mno  = 30e-3 / avgo
            mn4s = 14e-3 / avgo
        
        
        
            o2 = rho_tot* mmr_O2/mo2 #units are in meters-> divid
            n2 = rho_tot* mmr_N2/mn2
            o1 = rho_tot* mmr_O1/mo1
            no = rho_tot* mmr_NO/mno
            n4s = rho_tot* mmr_N4S/mn4s
        
            lon_idx = np.array([np.argmin(np.abs(lon - 33)), np.argmin(np.abs(lon - -128))])
            lat_idx = np.arange(len(lat))[((lat > -65)& (lat < 45))]
            ut_idx = np.arange(len(ut))
            samples = product(ut_idx, lat_idx, lon_idx)

            
            for ut_idx, lat_idx, lon_idx in samples:
                lt = ut[ut_idx,1] + lon[lon_idx] // 15
                dummy_z = Z[:,:,lat_idx, lon_idx]
                do2 = np.interp(170, dummy_z[ut_idx]/1e5, o2[ut_idx, :, lat_idx, lon_idx])
                data.append([ do2, lat[lat_idx], lon[lon_idx], lt, year, ut[ut_idx,1], ut[ut_idx,0], f107[ut_idx], kp[ut_idx]])
        except Exception:
            print('skipped')
            continue
    
    names = ['o2', 'lat', 'lon', 'lt', 'year', 'ut_hour', 'doy', 'f107', 'Kp']
    
    df = pd.DataFrame(data = data, columns = names )
    df['lt'] = df['lt'] %24
    df.to_csv(out)
    