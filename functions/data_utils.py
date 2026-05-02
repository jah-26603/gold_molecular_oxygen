# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 23:04:58 2025

@author: dogbl
"""

import pandas as pd
from astropy.time import Time

import glob
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

def f10_7_data(fp, start_date , end_date, threshold = 80, greater_than = True):

    df = pd.read_csv(fp)
    # --- Convert Julian Date to datetime ---
    jd = df['time (Julian Date)'].values
    dt = Time(jd, format='jd').to_value('iso', subfmt='date')  # removes time
    df['date'] = dt
    # --- Group by date and compute daily mean ---
    daily_df = (
        df.groupby('date')['adjusted_flux (solar flux unit (SFU))']
        .mean()
        .reset_index(name='daily_avg_sfu')
    )
    
    # --- Filter by date range and flux threshold ---
    if greater_than: 
        filtered_df = daily_df[
            (daily_df['date'] >= start_date) &
            (daily_df['date'] <= end_date) &
            (daily_df['daily_avg_sfu'] >= threshold)]
    else:
        filtered_df = daily_df[
            (daily_df['date'] >= start_date) &
            (daily_df['date'] <= end_date) &
            (daily_df['daily_avg_sfu'] < threshold)]
    return filtered_df




def collect_files_by_date_range(start_date, end_date, data_path='data/o2den'):
    """
    Collect NetCDF files within a specified date range.
    
    Parameters:
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format
    data_path (str): Base path to data directory
    
    Returns:
    list: Flattened list of file paths within the date range
    """
    s_dt = datetime.strptime(start_date, "%Y-%m-%d")
    e_dt = datetime.strptime(end_date, "%Y-%m-%d")
    years = np.arange(s_dt.year, e_dt.year + 1)
    s_doy = int(s_dt.strftime('%j'))
    e_doy = int(e_dt.strftime('%j'))
    
    file_list = []
    
    if len(years) > 1:
        for year in years:
            files = glob.glob(f'{data_path}/{year}/*.nc')
            
            if not files:  # Skip if no files found for this year
                continue
                
            # Extract day of year from filename (assuming format: *_DOYDOYDOY_*.nc)
            file_mask = []
            valid_files = []
            
            for file in files:
                try:
                    doy = int(file[-18:-15])  # Extract DOY from filename
                    file_mask.append(doy)
                    valid_files.append(file)
                except (ValueError, IndexError):
                    # Skip files that don't match expected naming convention
                    continue
            
            if not file_mask:
                continue
                
            file_mask = np.array(file_mask)
            valid_files = np.array(valid_files)
            
            if year == years[0]:  # First year: from start_doy to end of year
                doy_mask = file_mask >= s_doy
                file_list.extend(valid_files[doy_mask].tolist())
                
            elif year == years[-1]:  # Last year: from start of year to end_doy
                doy_mask = file_mask <= e_doy
                file_list.extend(valid_files[doy_mask].tolist())
                
            else:  # Middle years: all files
                file_list.extend(valid_files.tolist())
    
    else:  # Single year
        files = glob.glob(f'{data_path}/{years[0]}/*.nc')
        
        if files:
            file_mask = []
            valid_files = []
            
            for file in files:
                try:
                    doy = int(file[-18:-15])
                    file_mask.append(doy)
                    valid_files.append(file)
                except (ValueError, IndexError):
                    continue
            
            if file_mask:
                file_mask = np.array(file_mask)
                valid_files = np.array(valid_files)
                doy_mask = (file_mask >= s_doy) & (file_mask <= e_doy)
                file_list = valid_files[doy_mask].tolist()
    
    return sorted(file_list)  # Return sorted list for consistency


def resample_o2(data, **kwargs):
    #resample data into 2d image
    params = {
        'k_size': 21,
        'lt_list': np.linspace(0, 24, num=12, endpoint=False),
        'lat_list': np.linspace(-65, 50, num=23, endpoint=False),
        'doy_list': np.linspace(1, 365, num=12, endpoint=False),
        'plot': False
    }

    # Update with any user-supplied kwargs
    params.update(kwargs)

    # Access the parameters
    k_size = params['k_size']
    lt_list = params['lt_list']
    lat_list = params['lat_list']
    doy_list = params['doy_list']
    plot = params['plot']
        
    o2den_arr = [[[] for _ in lt_list] for _ in doy_list]
    o2den_unc_arr = [[[] for _ in lt_list] for _ in doy_list]
    l2den_arr = [[[] for _ in lat_list] for _ in doy_list]
    o2_3d = [[[[] for _ in lt_list] for _ in lat_list] for _ in doy_list]

    for i in range(len(data)):
        
        if data[i,2] > doy_list[-1]*( 1 + .5/(len(doy_list) - 1)): #closer to day 0
           doy_idx = 0
        else:
            doy_idx = np.argmin(np.abs(data[i,2] - doy_list))
        
        if data[i,1] > lt_list[-1] *( 1 + .5/(len(lt_list) - 1)):
            lt_idx = 0
        else:
            lt_idx = np.argmin(np.abs(data[i,1] - lt_list))
        lat_idx = np.argmin(np.abs(lat_list - data[i,3]))
        o2_3d[doy_idx][lat_idx][lt_idx].append(data[i,0])
        
    # Create 3D visualization
    o2_3d_img = np.array([[[np.nanmedian(o2_3d[doy][lat][lt]) if len(o2_3d[doy][lat][lt]) > 0 else np.nan 
                            for lt in range(len(lt_list))] 
                           for lat in range(len(lat_list))] 
                          for doy in range(len(doy_list))])
    std_3d_img = np.array([[[np.nanstd(o2_3d[doy][lat][lt]) if len(o2_3d[doy][lat][lt]) > 0 else np.nan 
                            for lt in range(len(lt_list))] 
                           for lat in range(len(lat_list))] 
                          for doy in range(len(doy_list))])
    if plot:
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))
        fig.suptitle('Data Availability', fontsize=16, y=0.98)
    
        from scipy.signal import medfilt
        
        # Sort data by Local Time for a proper line plot
        sorted_indices = np.argsort(data[:, 1])
        x_sorted = data[sorted_indices, 1]
        y_sorted = data[sorted_indices, 0]
        y_sorted = np.array(y_sorted, dtype=np.float32)
    
        # Existing scatter plot
        axes[0].scatter(data[:, 1], data[:, 0], alpha=0.6, s=30)
        
        # Median-filtered line plot (no new variable assigned)
        axes[0].plot(x_sorted, medfilt(y_sorted, kernel_size=k_size), color='black', linewidth=2, label='Median Filter')
        
        # Labels and formatting
        axes[0].set_ylim((1e8, 1e9))
        axes[0].set_xlim((0, 24))
        axes[0].set_xlabel('Local Time')
        axes[0].set_ylabel('O2 Density [cm^-3]')
        axes[0].set_title('O2 Density vs Local Time')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
          
    
        # Plot 2: DOY vs Local Time
        axes[1].scatter(data[:, 1], data[:, 2], alpha=0.6, s=30)
        axes[1].set_xlim((0, 24))
        axes[1].set_xlabel('Local Time')
        axes[1].set_ylabel('Day of Year')
        axes[1].set_title('Day of Year vs Local Time')
        axes[1].grid(True, alpha=0.3)
        
         
    
        # Plot 3: Latitude vs Local Time
        axes[2].scatter(data[:, 1], data[:, 3], alpha=0.6, s=30)
        axes[2].set_xlim((0, 24))
        axes[2].set_xlabel('Local Time')
        axes[2].set_ylabel('Latitude')
        axes[2].set_title('Latitude vs Local Time')
        axes[2].grid(True, alpha=0.3)
    
    
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)  # Make room for suptitle and legend
        plt.show()


    return o2_3d_img, std_3d_img


from astropy.convolution import interpolate_replace_nans, Gaussian2DKernel 
    
def conv_interpolate(image, **kwargs):
    # Default parameters
    params = {
        'k_std': 0.5,
        'title': 'filled',
        'x_list': np.linspace(1, 365, num=12, endpoint=False),
        'y_list': np.linspace(0, 24, num=12, endpoint=False),
        'x_label': 'doy',
        'y_label': 'lt',
        'cyclic_feature' : [1] #axis where a cyclic feature is, typically doy is a feature
    }

    # Update with any user-supplied kwargs
    params.update(kwargs)

    # Access parameters
    k_std = params['k_std']
    title = params['title']
    x_list = params['x_list']
    y_list = params['y_list']
    x_label = params['x_label']
    y_label = params['y_label']  
    cyclic_feature = params['cyclic_feature']

    kernel = Gaussian2DKernel(x_stddev= k_std)
    if len(cyclic_feature) != 0 :
        
        full_img = np.squeeze(np.array([image]))
        for axis in cyclic_feature:
            full_img = np.concatenate(np.squeeze([full_img]*3), axis=axis)
        
        img_fil = interpolate_replace_nans(full_img, kernel, boundary = 'extend')
        slices = []
        a, b = image.shape
        for i, dim in enumerate((a, b)):
            if i in cyclic_feature:
                start = dim
                end = 2 * dim
                slices.append(slice(start, end))
            else:
                slices.append(slice(None))
    
        img_fil = img_fil[tuple(slices)]
                
    else:
        img_fil = interpolate_replace_nans(image, kernel, boundary = 'extend')
    
    # get color limits from original image
    vmin = np.nanmin(image)
    vmax = np.nanmax(image)
    
    # # plot
    # plt.figure(figsize=(15, 6))
    # plt.suptitle(f'{title} vs {x_label} and {y_label}')
    # plt.subplot(1, 2, 1)
    # im1 = plt.pcolor(x_list, y_list, image, vmin=vmin, vmax=vmax)
    # plt.xlabel(x_label)
    # plt.ylabel(y_label)
    # # plt.ylim((-45,45))
    # plt.title('O2den')
    # plt.colorbar(im1, label = '[cm^-3]')
    

    # plt.subplot(1, 2, 2)
    # im3 = plt.pcolor(x_list, y_list, img_fil, vmin=vmin, vmax=vmax)
    # plt.xlabel(x_label)
    # plt.ylabel(y_label)
    # plt.title('Interpolated O2den')
    # plt.colorbar(im3, label = '[cm^-3]')
    # # plt.ylim((-45,45))
    # plt.tight_layout()
    # plt.show()

    return img_fil

    
    
# # Example usage:
# if __name__ == "__main__":
#     start_date = '2019-01-01'
#     end_date = '2021-07-31'
    
#     collected_files = collect_files_by_date_range(start_date, end_date)
#     f10_7_df = f10_7_data(r"C:\Users\dogbl\OneDrive\Desktop\England Research\O2Density\data\F10_7_Radio.csv.csv"
#                           , start_date, end_date)