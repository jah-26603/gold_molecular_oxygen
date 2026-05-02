# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 15:05:35 2025

@author: dogbl
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from tqdm import tqdm
import netCDF4 as nc
import statsmodels.api as sm
import datetime as dt
from pymsis.utils import get_f107_ap


def load_gold_o2(fp, all_alts = False):
    data = []
    
    for year in tqdm(os.listdir(fp)):
        files = glob.glob(os.path.join(fp, year, '*.nc'))
        
        for file in tqdm(files):
            
            ds = nc.Dataset(file, 'r')
            o2 = ds.variables['o2den'][:].data
            o2_lon = ds.variables['lon_ref'][:].data 
            o2_lat = ds.variables['lat_ref'][:].data
            o2_utc = np.array([''.join(cell.decode('utf-8') for cell in row if cell) for row in ds.variables['time_utc'][:]])
            o2_utc_dt = np.array([dt.datetime.strptime(t, '%Y-%m-%dT%H:%M:%S.%fZ') for t in o2_utc])
            alt = ds.variables['zret'][:].data
            sza = ds.variables['sza_ref'][:].data

            if not all_alts:
                for i in range(len(o2_lon)):
                    data.append([o2[i,16], o2_lon[i], o2_lat[i], o2_utc_dt[i], sza[i]])
            else:
                for i in range(o2.shape[0]):
                    for j in range(o2.shape[1]):
                        data.append([o2[i,j], o2_lon[i], o2_lat[i], o2_utc_dt[i], alt[j]])
                    
                
            ds.close()
    if all_alts:
        o2_df = pd.DataFrame(data = data, columns = ['o2', 'o2_lon', 'o2_lat', 'o2_datetime','o2_alt'])
    else:
        o2_df = pd.DataFrame(data = data, columns = ['o2', 'o2_lon', 'o2_lat', 'o2_datetime', 'o2_sza'])
    o2_df['date'] = pd.to_datetime(o2_df.o2_datetime).dt.date
    o2_df['doy'] = o2_df.o2_datetime.dt.dayofyear
    o2_df['o2_ut'] = pd.to_datetime(o2_df.o2_datetime).dt.hour + pd.to_datetime(o2_df.o2_datetime).dt.minute/60
    o2_df['o2_lt'] = (o2_df.o2_ut + o2_df.o2_lon//15 + 24)%24
    
    o2_df = o2_df[o2_df.o2_datetime < pd.Timestamp('2025-08-01 12:00:00')]
    f107, _, ap = get_f107_ap(o2_df.o2_datetime)
    o2_df['f107'] = f107
    o2_df['ap'] = ap[:,1]
    
    return o2_df


def load_icon_on2(fp):
    ds = np.load(fp)
    data_dict = {key: ds[key].squeeze() for key in ds.files}
    on2_df = pd.DataFrame(data_dict)
    on2_df['icondisktime'] = pd.to_datetime(on2_df.icondisktime)
    on2_df['date'] = pd.to_datetime(on2_df.icondisktime).dt.date
    
    return on2_df
    
def load_gold_on2(fp, max_year = 2022):
    data = []
    dirs = [d for d in os.listdir(fp) if os.path.isdir(os.path.join(fp,d))]
    for year in dirs:
        if int(year)> max_year:
            continue
        for file in tqdm(glob.glob(os.path.join(fp, year, '*.nc'))):
            ds = nc.Dataset(file, 'r')
            on2 = ds.variables['on2'][:]
            dqi = ds.variables['on2_dqi'][:]
            utc = ds.variables['scan_start_time'][:] #just use start time, idk
            lon = ds.variables['longitude'][:]
            lat = ds.variables['latitude'][:]
            
            datetimes = []
            for row in utc:
                clean = row.compressed()  # remove masked bytes
                if clean.size == 0:
                    datetimes.append(None)
                else:
                    s = b''.join(clean).decode('utf-8')
                    dd = dt.datetime.strptime(s, '%Y-%m-%dT%H:%M:%SZ')
                    datetimes.append(dd)
            
            
            utc = np.broadcast_to(np.array(datetimes)[:, None, None], on2.shape)
            lat = np.broadcast_to(lat[None,:,:], on2.shape)
            lon = np.broadcast_to(lon[None,:,:], on2.shape)
            
            
            on2x = on2.flatten()
            dqix = dqi.flatten()
            utcx = utc.flatten()
            latx = lat.flatten()
            lonx = lon.flatten()
            # Define mask
            mask = ((dqix == 0) & (latx < 70) & (latx > -75) & (~np.isnan(lonx)) )
                    # & (lonx - np.nanmin(lonx) < 45) & (np.nanmax(lonx) - lonx < 45))           
            

            # Apply mask to each array
            on2x = on2x[mask]
            dqix = dqix[mask]
            utcx = utcx[mask]
            latx = latx[mask]
            lonx = lonx[mask]
            
            # Stack only the filtered data
            stacked = np.column_stack([on2x, dqix, utcx, latx, lonx])
            data.append(stacked)
            ds.close()
    data = np.vstack(data)
    on2_df = pd.DataFrame(data = data, columns = ['on2', 'dqi', 'on2_datetime', 'on2_lat', 'on2_lon'])
    on2_df['date'] = pd.to_datetime(on2_df.on2_datetime).dt.date
    on2_df['on2_ut'] = pd.to_datetime(on2_df.on2_datetime).dt.hour + pd.to_datetime(on2_df.on2_datetime).dt.minute/60
    on2_df['on2_lt'] = (on2_df.on2_ut + on2_df.on2_lon//15 + 24) % 24
    
    return on2_df


def load_icon_limb(fp, match_gold = True):
    os.listdir(fp)
    data = []
    dd = []
    for year in tqdm(os.listdir(fp)):
        for orbit in tqdm(glob.glob(os.path.join(fp,year,'*.nc'))):
            
            ds = nc.Dataset(orbit,'r')
            # epoch = ds.variables['Epoch'][:]
            temperature = ds.variables['ICON_L24_Temperature'][:].data
            o2 = ds.variables['ICON_L24_O2'][:]
            n2 = ds.variables['ICON_L24_N2'][:]
            alt = ds.variables['ICON_L24_Altitude'][:]
            try:
                mask = (alt.mask == True).all(axis = 1)
            except np.exceptions.AxisError:
                mask = (alt == 0).all(axis = 1)
                print('no mask for altitude?')
            lon = ds.variables['ICON_L24_Longitude'][:].data
            lat = ds.variables['ICON_L24_Latitude'][:].data
            time_utc = ds.variables['ICON_L24_UTC_Time'][:]
    
            lon = lon[mask == False]
            lat = lat[mask == False]
            time_utc = time_utc[mask == False]
            o2 = o2[mask == False]
            n2 = n2[mask == False]
            alt = alt[mask== False]
            temperature = temperature[mask == False]
            
            lat = np.broadcast_to(lat[:,None], shape = o2.shape).flatten()
            lon = np.broadcast_to(lon[:,None], shape = o2.shape).flatten()
            time_utc = np.broadcast_to(time_utc[:,None], shape = o2.shape).flatten()
            
            data.append(np.column_stack((lon,lat,time_utc, alt.flatten(), o2.flatten(), n2.flatten(), temperature.flatten())))
            ds.close()
            
    
    data = np.vstack(data)
    df = pd.DataFrame(data = data, columns = ['icon_lon','icon_lat','icon_utc', 'icon_alt', 'icon_o2', 'icon_n2', 'temperature'])
    
    if match_gold:
        df = df[ (df.icon_lon > -150) & (df.icon_lon < 50)].reset_index(drop = True)
    df['icon_datetime'] = pd.to_datetime(df.icon_utc)
    df['date'] = pd.to_datetime(df.icon_utc).dt.date
    df['icon_ut'] = pd.to_datetime(df.icon_utc).dt.hour + pd.to_datetime(df.icon_utc).dt.minute/60
    df['icon_lt'] = (df.icon_ut + df.icon_lon//15 + 24) %24
    
    return df

def load_guvi_on2(fp, match_gold = True):
    from scipy import stats as st
    # fp = r'E:\on2\guvi_on2_data'
    data = []
    for year in tqdm(os.listdir(fp)):
        for file in tqdm(glob.glob(os.path.join(fp,year, '*.nc'))):
            ds = nc.Dataset(file,'r')
            
            lat = ds.variables['LATITUDE'][:].data
            lon = ds.variables['LONGITUDE'][:].data
            on2 = ds.variables['ON2'][:].data
            
            utc = (ds.variables['FRACTIONAL_DOY'][:].data%1) * 24
            
            year = np.array([ds.variables['YEAR'][:].data[0]] * len(lon))
            doy = np.array([int(file[-17:-14])] * len(lon))
            
            
            data.append(np.column_stack((lon, lat, utc, on2, doy,year)))
    
    data = np.vstack(data)
    df = pd.DataFrame(data = data, columns = ['guvi_lon','guvi_lat','guvi_ut','guvi_on2', 'doy', 'year'])
    if match_gold:
        df = df[ (df.guvi_lon > -150) & (df.guvi_lon < 50)].reset_index(drop = True)
    df['guvi_lt'] = (df.guvi_ut + df.guvi_lon//15 + 24) % 24
    df['guvi_datetime'] = df.apply( lambda row: dt.datetime(int(row['year']), 1, 1) + dt.timedelta(days=int(row['doy']) - 1, hours=row['guvi_ut']),axis=1)
    df['date'] = df.guvi_datetime.dt.date
    f107, _, ap = get_f107_ap(df.guvi_datetime)
    df['f107'] = f107
    df['ap'] = ap[:,1]
    
    
    return df

def nam_sam_idx(fp = r"C:\Users\JDawg\OneDrive\Desktop\England Research\O2Density\data\MERRA2_dTdy_Ubar_SSW_NAM+SAM_19800101-20240331_3d.nc"):

    ds = nc.Dataset(fp, 'r')
    
    print(ds.variables.keys())
    
    
    idx = np.where(ds.variables['PRESSURE'][:] == 10)
    
    date = ds.variables['DATE'][:]
    NAM = np.squeeze(ds.variables['NAM_INDEX_3d'][:][idx])
    SAM = np.squeeze(ds.variables['SAM_INDEX_3d'][:][idx])
    
    data = np.stack([date, NAM, SAM], axis = 1)
    df = pd.DataFrame(data = data, columns = ['dd', 'nam', 'sam'])
    df['year'] = df['dd'].apply(lambda x: str(x)[:4])
    df['month'] = df['dd'].apply(lambda x: str(x)[4:6])
    df['day'] = df['dd'].apply(lambda x: str(x)[6:])
    return df


def load_o2wNe(o2_df, fp = r'E:\on2\cosmic2_gis'):
    #get pairings with o2
    Ne_lon_list, Ne_lat_list, Ne_den_list, hmfe2 = [], [], [],[]
    alt = np.linspace(0, 1000, 51)
    lat = np.linspace(-90, 90, num=73)
    # lon = np.linspace(0, 360, num=72, endpoint=False)
    lon = np.linspace(-180, 180, num=72, endpoint=False)
    a = 2 # grid is 2*a + 1, 2*a +1 ... for a = 1 (3x3 grid)
    for _, row in tqdm(o2_df.iterrows(), total=len(o2_df)):
        year = int(row.year)
        doy = int(row.doy)
        ut = int(row.o2_ut)
        
        file = os.path.join(fp, f'GIS_Ne_IRI_RO_GPS_{year}_{str(doy).zfill(3)}_{str(ut).zfill(2)}.nc')
        
        
        if not os.path.exists(file):
            Ne_lon_list.append(np.nan)
            Ne_lat_list.append(np.nan)
            Ne_den_list.append(np.nan)
            hmfe2.append(np.nan)
            continue

        try:
            ds = nc.Dataset(file, 'r')
        except OSError:
            continue
        

        Ne = ds.variables['Ne'][:].data #1e5 cm^-3 
        


        
        # Ne = alt[np.argmax(Ne, axis = 0)]
        
        from scipy import ndimage
        # Ne = ndimage.median_filter(np.trapz(Ne[5:]*1e5*1e6,dx = 20*1e3, axis = 0), size=5)/1e16 #convert to SI and then normalized TEC
        Ne = ndimage.median_filter(np.trapz(Ne[8:11]*1e5*1e6,dx = 20*1e3, axis = 0), size=5)/1e16 #convert to SI and then normalized TEC
        # Ne = ndimage.median_filter(Ne[9], size=5)
        # breakpoint()
        glat = np.argmin(np.abs(row.o2_lat - lat))
        glon = np.argmin(np.abs(row.o2_lon - lon))  # wrap-around-safe
        lat_min = max(glat - a, 0)
        lat_max = min(glat + a + 1, len(lat))
        lon_min = max(glon - a, 0)
        lon_max = min(glon + a + 1, len(lon))
        
        lat_patch = lat[lat_min:lat_max]
        lon_patch = lon[lon_min:lon_max]
        
        Lon_grid, Lat_grid = np.meshgrid(lon_patch, lat_patch)


        Ne_patch = Ne[lat_min:lat_max, lon_min:lon_max]


        Ne_lon_list.append(Lon_grid.mean() % 360)
        Ne_lat_list.append(Lat_grid.mean())
        Ne_den_list.append(np.nanmedian(Ne_patch))
        hmfe2.append(np.nanmedian(alt[np.argmax(ds.variables['Ne'][:].data, axis = 0)][lat_min:lat_max, lon_min:lon_max]))
        
    o2_df['Ne_lon'] = Ne_lon_list
    o2_df['Ne_lat'] = Ne_lat_list
    o2_df['Ne_den'] = Ne_den_list      
    o2_df['Ne_ut'] = o2_df['o2_datetime'].dt.floor('h')
    o2_df['hmfe2'] = hmfe2
    # o2_df = o2_df.explode(['Ne_lon', 'Ne_lat','Ne_den']).reset_index()
    # o2_df['lon'] = o2_df.o2_lon%360   #This may need to occur
    return o2_df


    
    

    