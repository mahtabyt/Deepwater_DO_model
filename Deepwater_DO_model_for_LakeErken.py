# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 09:53:54 2025

@author: Mahtab Yaghouti 
"""
#%% This file has been used to simulate the deepwater dissolved oxygen (DO) profile in 
# Lake Erken from 2019 to 2022:
    
# You are required to save the "input-data" files in your own input directory     
    

from pathlib import Path

#=== Define the input directory with the full path  
input_dir = Path(r'C:\Users\YOUR_USERNAME\path\to\your\input-data')  # <-- CHANGE THIS!
#===The input files were read as ===

# 1) bathymetry_file_Erken, 
# 2) temp_daily_prof_Erken 
# 3) DO_daily_prof_Erken
#===================================


#=== Cautions:=====================
# 1) Density calculation is based on a freshwater lake with a negligible amount of salt
# 2) For oxygen saturation functions change the "altitude_m" value based on your case 
# 3) unites for DO [mg/L] , temp [°C] and kz [m2/s]
#==================================



#=== Define an output directory with the full path to save the results 
output_dir = Path(r'C:\Users\YOUR_USERNAME\path\to\your\output-data')  # <-- CHANGE THIS!

#%%1) Reading the bathymetry data
# This script reads lake bathymetry data and calculates volume and surface area for depth layers.
# Users can replace the file path and layer segmentation to adapt it for other lakes.

import pandas as pd
import numpy as np
import scipy.interpolate
import os

# === Set working directory and input file ===
os.chdir(input_dir)

# Example: loading your bathymetry dataframe (replace this with your actual file path or df)
df_bathy_Erken = pd.read_csv('bathymetry_file_Erken.csv')

# Assuming your dataframe looks like this:
# Columns: ['Z(m)', 'A(m2)', 'V(m3)']
# Make sure it's sorted from surface (shallowest) to deepest (most negative Z)

df_bathy_Erken['V_cum'] = df_bathy_Erken['V(m3)'][::-1].cumsum()[::-1]
df_bathy_Erken['A_cum'] = df_bathy_Erken['A(m2)'][::-1].cumsum()[::-1]


#%%2) Reading the daily temperature profile 

# === Set working directory and input file ===
os.chdir(input_dir)

# === Read example data for Lake Erken ===
temp_daily_prof_Erken_2019_2022 = pd.read_csv('temp_daily_prof_Erken_2019_2022.csv', parse_dates=['Datetime'])

# === Preview the data ===
print(temp_daily_prof_Erken_2019_2022.head())
"""
            Z_m+      Temp
Datetime                  
2019-04-17   1.0  3.641818
2019-04-17   1.5  3.642727
2019-04-17   2.0  3.647273
2019-04-17   2.5  3.647273
2019-04-17   3.0  3.643636
"""

#%%3) Reading the daily DO profiles (after preprocessing) 

# === Set working directory and input file ===
os.chdir(input_dir)

# === Read example data for Lake Erken ===
DO_daily_prof_Erken_2019_2022 = pd.read_csv('DO_daily_prof_Erken_2019_2022.csv', parse_dates=['Datetime'])

# === Preview the data ===
print(DO_daily_prof_Erken_2019_2022.head())

"""
            Z_m+         DO
Datetime                   
2019-04-17   1.0  15.374891
2019-04-17   1.5  15.371987
2019-04-17   2.0  15.371019
2019-04-17   2.5  15.365210
2019-04-17   3.0  15.358434

"""


#%% 4) Create morphometry file aligned with temperature and DO profile Layers

# This section interpolates the bathymetric data of Lake Erken to a finer vertical resolution (0.5 m),
# matching the segmentation used in observed daily temperature and dissolved oxygen profiles (2019–2022).


# === Input: Bathymetric cumulative area and volume data ===
# Z_b, A_bath_cum, and V_bath_cum should be extracted from df_bathy_Erken

Z_b = df_bathy_Erken['Z(m)'].values
A_bath_cum = df_bathy_Erken['A_cum'].values
V_bath_cum = df_bathy_Erken['V_cum'].values


# === Define vertical segmentation (0.5 m resolution) from surface to maximum depth ===

vertical_resolution_profile= 0.5 # For Erken file  (adjust it for your lake) 
Z_up_bndr_m = np.append(0, np.arange(-1.25, -17.25, -1* vertical_resolution_profile))  # Upper boundaries of each layer the range should be adjusted to lake of interest 
Z_lo_bndr_m = np.append(np.arange(-1.25, -17.25,-1* vertical_resolution_profile), -21)  # Lower boundaries of each layer the range should be adjusted to lake of interest 

# ==== Interpolate bathymetric functions ===
As_interp = scipy.interpolate.interp1d(Z_b, A_bath_cum, bounds_error=False, fill_value="extrapolate")
Vcum_interp = scipy.interpolate.interp1d(Z_b, V_bath_cum, bounds_error=False, fill_value="extrapolate")

As_up_bndr_m = As_interp(Z_up_bndr_m)      # Interpolated surface area at upper boundary
As_lo_bndr_m = As_interp(Z_lo_bndr_m)      # Interpolated surface area at lower boundary
Vcum_up_bndr = Vcum_interp(Z_up_bndr_m)    # Interpolated cumulative volume at upper boundary
Vcum_lo_bndr = Vcum_interp(Z_lo_bndr_m)    # Interpolated cumulative volume at lower boundary

# ==== Calculate layer-wise volume and lateral area ====
V_r_m = Vcum_up_bndr - Vcum_lo_bndr        # Layer volume [m³]
A_l_m = As_up_bndr_m - As_lo_bndr_m        # Lateral sediment contact area [m²]
Z_m = np.arange(-1, -17.5, -1* vertical_resolution_profile)           # Midpoint of each layer

# ==== Layer thickness ====
thickness_up_lo_of_m = np.append(-1 * np.diff(Z_m), 21 - 16.75)  # Bottom layer thickness to -21 m
thickness_up_lo_of_m[0] = 1.25  # Set top layer thickness explicitly

# ==== Construct morphometry DataFrame ====
df_morphrometry = pd.DataFrame({
    'Z_m': Z_m,
    'thickness': thickness_up_lo_of_m,
    'V_r_m': V_r_m,
    'A_l_m': A_l_m,
    'As_up_bndr_m': As_up_bndr_m,
    'As_lo_bndr_m': As_lo_bndr_m,
    'Vcum_m': Vcum_lo_bndr
})


# === Save to CSV in output directory ====
df_morphrometry.to_csv(output_dir / 'processed_bathymetry.csv', index=False)



# A list of all useful hypsographic information in full profile including:
# 1) volume of each layer, 2) surface area at the upper boundary of each layer 
# 3) center of each layer depth from surface and the 4) vertical resolution in temperature measurement
full_prof_morpho_Erken= [np.array (df_morphrometry['V_r_m'][1:]) ,
                                     np.array(df_morphrometry['As_up_bndr_m'][1:]) ,
                                     np.array(df_morphrometry['Z_m'][1:]) , 
                                     vertical_resolution_profile ]


# A list of all useful hypsographic information in deepwater profile: 
# 1) volume of each layer, 2) surface area at the upper boundary of each layer 
# 3) the sediment area or lateral area of each layer and 4) the vertical resolution in temperature measurement
hypo_prof_morpho_Erken= [np.array (df_morphrometry[df_morphrometry['Z_m']<-13.5]['V_r_m']) ,
                                     np.array(df_morphrometry[df_morphrometry['Z_m']<-13.5]['As_up_bndr_m']) ,
                                     np.array(df_morphrometry[df_morphrometry['Z_m']<-13.5]['A_l_m']) , 
                                     vertical_resolution_profile ]





#%%5) Density profile calculations from temperature data 
# in freshwater with marginal salinity 
# following Gill, 1982; Imberger & Patterson, 1989; Millero & Poisson, 1981 

def calculate_density(Temp):
    return (999.842594+6.793952*10**-2*Temp-9.095290*10**-3*Temp**2 + 1.001685*10**-4*Temp**3-1.120083*10**-6 * Temp**4+6.536332 * 10**-9*Temp**5)
       

 


#%%6) Stratification onset and breakdown identification  

#===The two functions of "consecutive_criteria" and "find_consecutive_ranges"
#===were identified to make sure the threshold of stratification duration will meet 
#===for several consecutive days (in Erken it was 7 days)


# From this function, we create a boolean type array that checks the meeting of a threshold for several consecutive days 
def consecutive_criteria(density_difference, threshold , consecutive_days):
    criteria = None
    for i in range(consecutive_days):
        shift_data = density_difference.shift(-i)
        condition = shift_data >= threshold
        if criteria is None:
            criteria = condition
        else:
            criteria = criteria & condition
    return criteria
	

# Function for finding subsets of days when they meet the condition 
def find_consecutive_ranges(dates , min_duration):
    onset_dates=[]
    offset_dates = []
    start_date = None
    end_date = None

    for date in dates:
        if end_date is None or (date - end_date).days == 1:
            end_date = date
            if start_date is None:
                start_date = date
        else:
            if start_date is not None and end_date is not None and (end_date - start_date).days >=min_duration:
                onset_dates.append(start_date)
                offset_dates.append(end_date)
                
            start_date = date
            end_date = date
    
    if start_date is not None and end_date is not None and (end_date - start_date).days >= min_duration:
        onset_dates.append(start_date)
        offset_dates.append(end_date)
    return onset_dates, offset_dates





def find_str_period(dens_diff_bnd_surf, temp_surf, temp_surf_lower_threshold=4, dens_diff_threshold= 0.05, consecutive_days_num=7):
    
    #===== The inputs are:=====
    #1. dens_diff_bnd_surf= The data series indexed by date hold the daily density difference between the layer above the deepwater (boundary layer) and surface layer 
    #2. temp_surf= The data series indexed by date of surface temperature  
    #3. temp_surf_lower_threshold= A threshold for identifying summer stratification by having warmer surface temperature than 4°C
    #4. dens_diff_threshold=  Density thershold for stratification identification
    #5. consecutive_days_num= The number of consecutive days for meeting the criteria for the sake of persistency  
    #===== The output is:=====
    
    #df_info_str: a dataframe has all the necessary information of a stratified period in one row 
    #within 11 columns including:
    
    # 'index_str_period_in_year'
    # 'onset_dates' : Stratification onset date
    # 'onset_indices': The index number of stratification onset date
    # 'initial_cond_indices': The index number of initial condition dates (a day before stratification onset) 
    # 'end_dates' : Stratification breakdown date
    # 'end_indices': The index number of stratification breakdown date
    # 'str_duration': The number of stratification duration 
    # 'replenishment_duration' : The number of replenishment duration 
    # 'end_dates_Jday': The turnover day of the year (turnover julian date)
    # 'onset_dates_Jday': The onset day of the year  (onset Julian date)
    # 'year': The year number 
    
    
    
    # Removing those days with the surface temperature of less than 4°C:
    df_surf_temp = pd.DataFrame({'temp_surf': temp_surf})
    df_surf_temp = df_surf_temp.set_index(dens_diff_bnd_surf.index)

    dens_diff_allyears_non_index = dens_diff_bnd_surf.reset_index()

    onset_dates = []
    end_dates = []

    onset_indices = []
    end_indices = []

    str_duration = []
    replenishment_duration = []

    for year in range(dens_diff_bnd_surf.index.year.min(), dens_diff_bnd_surf.index.year.max() + 1):
        dens_diff_oneyear = dens_diff_bnd_surf.loc[dens_diff_bnd_surf.index.year == year]
        df_surf_temp_oneyear = df_surf_temp.loc[df_surf_temp.index.year == year]

        criteria = consecutive_criteria(dens_diff_oneyear, dens_diff_threshold, consecutive_days_num)
        true_criteria_dates_density = dens_diff_oneyear[criteria].index

        true_criteria_dates_temp = df_surf_temp_oneyear[df_surf_temp_oneyear['temp_surf'] > temp_surf_lower_threshold].index

        # combination of temp threshold for surface, and density diff for consecutive days
        true_criteria_dates = true_criteria_dates_temp.intersection(true_criteria_dates_density)

        str_onset, str_end = find_consecutive_ranges(true_criteria_dates, consecutive_days_num)

        onset_dates.extend(str_onset)
        end_dates.extend(str_end)

    for s, e in zip(onset_dates, end_dates):
        onset_index = np.where(dens_diff_allyears_non_index['Datetime'] == s)[0][0]
        end_index = np.where(dens_diff_allyears_non_index['Datetime'] == e)[0][0]

        onset_indices.append(onset_index)
        end_indices.append(end_index)

    str_duration_deltas = np.array(end_dates) - np.array(onset_dates)
    str_duration = [delta.days + 1 for delta in str_duration_deltas]

    replan_duration_deltas = np.array(onset_dates[1:]) - np.roll(np.array(end_dates), 1)[1:]
    replenishment_duration = [np.nan] + [delta.days - 1 for delta in replan_duration_deltas]

    # Extract the year from each date and count occurrences
    years = [date.year for date in onset_dates]

    # Create a dictionary to store counts
    year_counts = {}

    # Create an array with the count of each year
    index_deox_period_in_year = []

    for year in years:
        if year in year_counts:
            year_counts[year] += 1
        else:
            year_counts[year] = 1
        index_deox_period_in_year.append(year_counts[year] - 1)

    # Create a dictionary with the collected data
    data_info_str = {
        'index_str_period_in_year': index_deox_period_in_year, # Determine the number of stratification in a year (startarted with 0) 
        'onset_dates': onset_dates, # Stratification onset date
        'onset_indices': onset_indices, # The index number of stratification onset dates
        'initial_cond_indices' : np.array(onset_indices)- 1 , # The index number of initial condition dates which is a day before stratification starts 
        'end_dates': end_dates, # Stratification breakdown dates
        'end_indices': end_indices,# The index number of stratification breakdown dates
        'str_duration': str_duration,# The number of stratification duration 
        'replenishment_duration': replenishment_duration # The number of replenishment duration days
    }
    
    

    # Create a DataFrame
    df_info_str = pd.DataFrame(data_info_str)
    
    #replenishment is only required for summertime mixing events
    df_info_str.loc[df_info_str['index_str_period_in_year'] == 0, 'replenishment_duration'] = np.nan

    df_info_str['onset_dates'] = pd.to_datetime(df_info_str['onset_dates'])
    df_info_str['end_dates'] = pd.to_datetime(df_info_str['end_dates'])

    df_info_str['end_dates_Jday'] = df_info_str['end_dates'].dt.dayofyear # The turnover day of the year (turnover Julian date)
    df_info_str['onset_dates_Jday'] = df_info_str['onset_dates'].dt.dayofyear # The onset day of the year  (onset Julian date)
    df_info_str['year'] = df_info_str['onset_dates'].dt.year
    
    
    return df_info_str



#%% 7) Using the function of "find_stratification_period" in 2019-2022 data in Lake Erken 

#==== Set index of Datetime column ====
temp_daily_prof_Erken_2019_2022= temp_daily_prof_Erken_2019_2022.set_index(['Datetime'])


#==== Calculate the density difference between the boundary layer ( Z_m+ = 13.5 m)
#==== and the surface water ( Z_m+ = 1 m) ====


#Temperature at boundary layer and surface layer 
temp_bnd_Erken_2019_2022= temp_daily_prof_Erken_2019_2022[temp_daily_prof_Erken_2019_2022['Z_m+']==13.5]['Temp']

temp_surf_Erken_2019_2022= temp_daily_prof_Erken_2019_2022[temp_daily_prof_Erken_2019_2022['Z_m+']==1.0]['Temp']


#==== Density at boundary layer and surface layer ==== 


density_bnd_Erken_2019_2022= calculate_density (temp_bnd_Erken_2019_2022 )

density_surf_Erken_2019_2022= calculate_density (temp_surf_Erken_2019_2022 )


#==== Density difference between the boundary layer and surface layer ==== 

density_diff_Erken_2019_2022= density_bnd_Erken_2019_2022 - density_surf_Erken_2019_2022



#=== Estimating a dataframe for stratification information in Erken  

df_str_info_Erken_2019_2022= find_str_period(density_diff_Erken_2019_2022, temp_surf_Erken_2019_2022, temp_surf_lower_threshold=4, dens_diff_threshold= 0.05, consecutive_days_num=7)


# === Preview the data ===
print(df_str_info_Erken_2019_2022)

"""
   index_str_period_in_year onset_dates  ...  onset_dates_Jday  year
0                         0  2019-05-15  ...               135  2019
1                         1  2019-07-12  ...               193  2019
2                         0  2020-05-22  ...               143  2020
3                         0  2021-05-13  ...               133  2021
4                         0  2022-05-23  ...               143  2022

[5 rows x 11 columns]
"""

# === Save to CSV in output directory ====
df_str_info_Erken_2019_2022.to_csv(output_dir / 'df_str_info_Erken_2019_2022.csv', index=False)



#%% 8) Subsetting the temperature and DO profiles during the stratified period 


def subset_profiles_by_strat_period(prof_df, strat_df, prof_name):
    """
    Subset a profile DataFrame by stratification periods, 
    and return a dictionary of the subsets.

    Parameters
    ----------
    prof_df : pd.DataFrame
        Profile dataframe indexed by Datetime (e.g., temperature or DO).
    strat_df : pd.DataFrame
        Dataframe with columns 'onset_dates', 'end_dates',
        'year', and 'index_str_period_in_year'.
    prof_name : str
        Base name to include in each output file (e.g., 'temp', 'DO').

    Returns
    -------
    dict
        Dictionary of subset DataFrames keyed by names like
        '{prof_name}_str_{year}_{period_idx}'.
    """

    subset_dict = {}

    for _, row in strat_df.iterrows():
        start_date   = row['onset_dates']
        end_date     = row['end_dates']
        year         = row['year']
        period_idx   = row['index_str_period_in_year']

        # Subset for the stratification period
        subset = prof_df.loc[(prof_df.index >= start_date) &
                             (prof_df.index <= end_date)]

        # Name for dictionary key 
        key = f"{prof_name}_str_{year}_{period_idx}"

        # Store in the dictionary
        subset_dict[key] = subset



    return subset_dict


#=========== Using the subset function for temperature data 

temp_subset_dict= subset_profiles_by_strat_period (temp_daily_prof_Erken_2019_2022 , df_str_info_Erken_2019_2022 , 'temp'  )


#=======Create a dataframe for saving daily temperature profile for each stratified period 

df_info_prof_str_2019_0 = temp_subset_dict['temp_str_2019_0'].copy()
df_info_prof_str_2019_1 = temp_subset_dict['temp_str_2019_1'].copy()
df_info_prof_str_2020_0 = temp_subset_dict['temp_str_2020_0'].copy()
df_info_prof_str_2021_0 = temp_subset_dict['temp_str_2021_0'].copy()
df_info_prof_str_2022_0 = temp_subset_dict['temp_str_2022_0'].copy()


#=========== Using the subset function for DO data 

#== Set index of Datetime column ==
DO_daily_prof_Erken_2019_2022= DO_daily_prof_Erken_2019_2022.set_index(['Datetime'])


DO_subset_dict= subset_profiles_by_strat_period (DO_daily_prof_Erken_2019_2022 , df_str_info_Erken_2019_2022 , 'DO'  )


#=======Add values of oxygen data in the dataframe of profile information in each stratified period 

df_info_prof_str_2019_0['DO'] = DO_subset_dict['DO_str_2019_0']['DO']
df_info_prof_str_2019_1['DO'] = DO_subset_dict['DO_str_2019_1']['DO']
df_info_prof_str_2020_0['DO'] = DO_subset_dict['DO_str_2020_0']['DO']
df_info_prof_str_2021_0['DO'] = DO_subset_dict['DO_str_2021_0']['DO']
df_info_prof_str_2022_0['DO'] = DO_subset_dict['DO_str_2022_0']['DO']

#%% 9) Function for reshaping a dataframe into a 2D array (time x depth)
###=== , later on, will be used for calculation of mixing coefficient (Kz) values 

def two_D_array_from_df(df, values_name, index_name='Datetime', column_name='Z_m+'):
    """
    Pivot a dataframe to create a 2D array with time as rows and depth as columns.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        values_name (str): Name of the column to fill values in the 2D array (e.g. 'Temp' or 'DO').
        index_name (str): Column to use as the row index (default: 'Datetime').
        column_name (str): Column to use as the column headers (default: 'Z_m').

    Returns:
        A 2D array of values in the shape of (time, depth).
    """

    # Pivot the dataframe into a wide format
    pivoted = df.pivot_table(index=index_name, columns=column_name, values=values_name)

    # Sort columns from surface to bottom if depth values are negative
    if df[column_name].min() < 0:
        pivoted = pivoted.sort_index(axis='columns', ascending=False)

    return pivoted.to_numpy()



#=====useage of two_D_array_from_df for stratified temperature profiles 


temp_2D_str_2019_0= two_D_array_from_df(temp_subset_dict['temp_str_2019_0'], 'Temp', index_name='Datetime', column_name='Z_m+')

temp_2D_str_2019_1= two_D_array_from_df(temp_subset_dict['temp_str_2019_1'], 'Temp', index_name='Datetime', column_name='Z_m+')

temp_2D_str_2020_0= two_D_array_from_df(temp_subset_dict['temp_str_2020_0'], 'Temp', index_name='Datetime', column_name='Z_m+')

temp_2D_str_2021_0= two_D_array_from_df(temp_subset_dict['temp_str_2021_0'], 'Temp', index_name='Datetime', column_name='Z_m+')

temp_2D_str_2022_0= two_D_array_from_df(temp_subset_dict['temp_str_2022_0'], 'Temp', index_name='Datetime', column_name='Z_m+')
    
#=====These dataframes will be used for estimating the Kz profile 

#%% 10) Calculate vertical diffusivity (Kz) from temperature profiles


# This method estimates vertical diffusivity (Kz) from daily temperature data in unit of m2/s 
# Note: Kz cannot be calculated for the first day (in time) and the topmost layer (in depth),
# because of the need for temporal and vertical gradients.


# Required inputs:
# - temp_daily_2D_str_y_n: 2D numpy array of temperature (time x depth)
# - dt: 1D array of time step durations (in days)
# - full_prof_morpho_variables: tuple containing:
#     * V_r: volume between depth layers
#     * A_s: horizontal area at layer boundaries
#     * Z_m: depth levels
#     * dz: thickness of vertical layers

def calculate_Kz(temp_daily_2D_str_y_n, dt, full_prof_morpho=full_prof_morpho_Erken ):
    
    
    V_r, A_s, Z_m, dz = full_prof_morpho

    # Compute the rate of temperature change over time (excluding first time-step and top layer)
    a = np.diff(temp_daily_2D_str_y_n[:, 1:], axis=0) / (dt[:, np.newaxis] * 24 * 3600)

    # Compute the vertical temperature gradient at upper boundary of each box (excluding the bottom layer and first day)
    G = -1 * (np.diff(temp_daily_2D_str_y_n, axis=1) / dz)[:-1, :]

    # Initialize Kz array (one element fewer in time and depth)
    kz = np.zeros([temp_daily_2D_str_y_n.shape[0] - 1, temp_daily_2D_str_y_n.shape[1] - 1])

    # Loop over each time step
    for time in range(temp_daily_2D_str_y_n.shape[0] - 1):
        # Loop over each depth layer
        for index, _ in enumerate(Z_m):
            # Calculate Kz using the integral heat budget approximation
            kz[time, index] = (np.dot(a[time, index:], V_r[index:])) / (G[time, index] * A_s[index])

    
    return kz



#%%11) For calculating the fortnightly Kz we need fortnightly temperature profiles  
# The periodic average temperature profiles can be identified based on the desired intervals (e.g., 7 , 14 , or 30 days)
# The function also returns the duration of each period and the average spacing between periods.


def calculate_period_average(data_slice):
    """
    Calculate the average temperature profile over a given time slice.
    
    Parameters:
        data_slice (np.ndarray): A 2D array (days x depths) representing temperature data for a period.

    Returns:
        np.ndarray: The mean temperature profile for the period.
    """
    return np.mean(data_slice, axis=0)


def calculate_periodic_averages(temperature_2D_data, days_per_period):
    """
    Divide a temperature time series into periods and calculate average profiles for each.

    Parameters:
        temperature_data: 2D array (days x depths) of daily temperature profiles.
        days_per_period (int): Number of days to average over for each period (e.g., 7, 14, 30).

    Returns:
        tuple:
            - averages: 2D array of average temperature profiles per period.
            - duration_total: 1D array of the actual duration (in days) for each period.
            - difference_between_periods: 1D array of average temporal spacing between periods.
    """
    num_days = len(temperature_2D_data)
    num_periods = num_days // days_per_period
    remaining_days = num_days % days_per_period

    averages = []          # List to collect average profiles
    duration_total = []    # List to collect durations

    # Loop through complete periods
    for i in range(num_periods):
        start_day = i * days_per_period
        end_day = start_day + days_per_period
        avg_temp = calculate_period_average(temperature_2D_data[start_day:end_day])
        averages.append(avg_temp)
        duration_total.append(days_per_period)

    # Handle the final incomplete period (if any)
    if remaining_days > 0:
        start_day = num_periods * days_per_period
        avg_temp = calculate_period_average(temperature_2D_data[start_day:])
        averages.append(avg_temp)
        duration_total.append(remaining_days)

    # Convert lists to arrays
    averages = np.array(averages)
    duration_total = np.array(duration_total)

    # Calculate the average spacing (in days) between consecutive periods
    difference_between_periods = (duration_total[1:] + duration_total[:-1]) / 2

    return averages, duration_total, difference_between_periods


#%%12) Usage of calculate_periodic_averages function to have fortnightly 2-D array of temperature profiles during the stratified period 


#str_2019_0:
temp_fortnightly_2D_2019_0, duration_fortnightly_total_2019_0, difference_fortnightly_periods_2019_0= calculate_periodic_averages(temp_2D_str_2019_0 ,14 )
#str_2019_1:
temp_fortnightly_2D_2019_1, duration_fortnightly_total_2019_1, difference_fortnightly_periods_2019_1= calculate_periodic_averages(temp_2D_str_2019_1 ,14 )
#str_2020_0:
temp_fortnightly_2D_2020_0, duration_fortnightly_total_2020_0, difference_fortnightly_periods_2020_0= calculate_periodic_averages(temp_2D_str_2020_0 ,14 )
#str_2021_0:
temp_fortnightly_2D_2021_0, duration_fortnightly_total_2021_0, difference_fortnightly_periods_2021_0= calculate_periodic_averages(temp_2D_str_2021_0 ,14 )
#str_2022_0:
temp_fortnightly_2D_2022_0, duration_fortnightly_total_2022_0, difference_fortnightly_periods_2022_0= calculate_periodic_averages(temp_2D_str_2022_0 ,14 )


#%%13) Calculate the Kz fortnightly profile for Lake Erken during the stratified period 

#str_2019_0:
kz_fortnightly_2D_2019_0= calculate_Kz (temp_fortnightly_2D_2019_0, difference_fortnightly_periods_2019_0 )
#str_2019_1:
kz_fortnightly_2D_2019_1= calculate_Kz (temp_fortnightly_2D_2019_1, difference_fortnightly_periods_2019_1 )
#str_2020_0:
kz_fortnightly_2D_2020_0= calculate_Kz (temp_fortnightly_2D_2020_0, difference_fortnightly_periods_2020_0 )
#str_2021_0:
kz_fortnightly_2D_2021_0= calculate_Kz (temp_fortnightly_2D_2021_0, difference_fortnightly_periods_2021_0 )
#str_2022_0:
kz_fortnightly_2D_2022_0= calculate_Kz (temp_fortnightly_2D_2022_0, difference_fortnightly_periods_2022_0 )

# There might be some negative values at the surface but we are using the deep region values and 
# the minimum value will be set for 1.4e-7 in the next section 

#%% 14) Repeating fortnightly Kz values for daily resolution and setting a minimum value for Kz based on 


def convert_fortnightly_kz_to_daily_resolution(kz_fortnightly_2D, duration_fortnightly_total, days_per_period=14, kz_min_threshold=1.4e-7):
    """
    Convert a 2D array of fortnightly Kz values into a daily-resolution array by:
    - Repeating each profile for `days_per_period` days
    - Padding the end by repeating the last profile for the sum of the last two durations
    - Applying a minimum Kz threshold

    Parameters:
    ------------
    kz_fortnightly_2D : 
        2D array of Kz values (time x depth), one row per fortnight
    duration_fortnightly_total : 
        1D array of duration (in days) for each fortnightly period
    days_per_period : int, optional
        Number of days each profile represents (default is 14)
    kz_min_threshold : float, optional
        Minimum allowed Kz value (default is 1.4e-7)

    Returns:
    --------
    kz_daily_2D : 
        2D array of Kz values with daily resolution
    """
    # Repeat each row (except last) for the specified period
    kz_repeated = np.repeat(kz_fortnightly_2D[:-1, :], days_per_period, axis=0)

    # Repeat the last row to pad the end
    last_row = kz_fortnightly_2D[-1]
    size_of_repeat = duration_fortnightly_total[-1] + duration_fortnightly_total[-2]
    extra_rows = np.tile(last_row, (int(size_of_repeat), 1))

    # Stack the arrays to build full daily-resolution Kz
    kz_daily_2D = np.vstack((kz_repeated, extra_rows))

    # Enforce minimum Kz threshold
    kz_daily_2D[kz_daily_2D < kz_min_threshold] = kz_min_threshold

    return kz_daily_2D



#str_2019_0:
kz_fort_2019_0_daily = convert_fortnightly_kz_to_daily_resolution(kz_fortnightly_2D_2019_0, duration_fortnightly_total_2019_0)
#str_2019_1:
kz_fort_2019_1_daily = convert_fortnightly_kz_to_daily_resolution(kz_fortnightly_2D_2019_1, duration_fortnightly_total_2019_1)
#str_2020_0:
kz_fort_2020_0_daily = convert_fortnightly_kz_to_daily_resolution(kz_fortnightly_2D_2020_0, duration_fortnightly_total_2020_0)
#str_2021_0:
kz_fort_2021_0_daily = convert_fortnightly_kz_to_daily_resolution(kz_fortnightly_2D_2021_0, duration_fortnightly_total_2021_0)
#str_2022_0:
kz_fort_2022_0_daily = convert_fortnightly_kz_to_daily_resolution(kz_fortnightly_2D_2022_0, duration_fortnightly_total_2022_0)


#%% Assign kz values to temperature dataframe in the stratified profile information next to temperature and oxygen data

#2019_0
df_info_prof_str_2019_0.loc[
    (df_info_prof_str_2019_0['Z_m+'] > 1),  
    'kz']= kz_fort_2019_0_daily.ravel()

df_info_prof_str_2019_0.to_csv(output_dir / 'df_info_prof_str_2019_0.csv')

#2019_1
df_info_prof_str_2019_1.loc[
    (df_info_prof_str_2019_1['Z_m+'] > 1),  
    'kz']= kz_fort_2019_1_daily.ravel()

df_info_prof_str_2019_1.to_csv(output_dir / 'df_info_prof_str_2019_1.csv')

#2020_0
df_info_prof_str_2020_0.loc[
    (df_info_prof_str_2020_0['Z_m+'] > 1),  
    'kz']= kz_fort_2020_0_daily.ravel()

df_info_prof_str_2020_0.to_csv(output_dir / 'df_info_prof_str_2020_0.csv')

#2021_0
df_info_prof_str_2021_0.loc[
    (df_info_prof_str_2021_0['Z_m+'] > 1),  
    'kz']= kz_fort_2021_0_daily.ravel()

df_info_prof_str_2021_0.to_csv(output_dir / 'df_info_prof_str_2021_0.csv')

#2022_0
df_info_prof_str_2022_0.loc[
    (df_info_prof_str_2022_0['Z_m+'] > 1),  
    'kz']= kz_fort_2022_0_daily.ravel()

df_info_prof_str_2022_0.to_csv(output_dir / 'df_info_prof_str_2022_0.csv')


#%% Several functions were identified and used in deepwater oxygen modelling including:
    
    #1) "K_temp": calculate the temperature modifier of the oxygen depletion rate 
    #2) "C_satur" and "estimate_satur_percentage": related to the estimation of oxygen concentration, saturation level and temperature 
    #3) "daily_C_bnd": calculate oxygen concentration at the boundary layer during the stratified period 

#%% Identifying a function for calculating temperature modifier of oxygen depletion rates 


def K_temp(temp):
    theta = 1.087
    Ktemp=theta ** (temp - 20)
    return (Ktemp)



#%% Functions for relating oxygen saturation, temperature and oxygen concentration  

    
def C_satur ( temp, satur_percent, altitude_m=10 ):
    #=======inputs:========
    # temp: temperature value 
    # satur_percent: Saturation percentage in float number ranged between 0 (0%) to 1(fully saturated or 100%)
    # altitude_m: Lake altitude in m (for lake Erken it is 10 m )
    #=======outputs:=======
    # C_at_satur: DO concentration at identified saturation percentage (satur_percent)
    
    altitude_km = altitude_m/1000 
    P = np.exp(5.25 * np.log(1 - altitude_km/44.3))
    t = temp
    T = 273.15 + t
    teta = 0.000975 - (t*1.426*10**-5) + ((t**2)*6.436*10**-8)
    Pwv = 11.8571 - (3840.70/T)-(216961/T**2)
    C_star =np.exp(7.7117 - 1.31403 * np.log(t+45.93))
    C_at_satur= satur_percent*(C_star * P*(((1-(Pwv/P))*(1-teta*P))/((1-Pwv)*(1-teta))))  
    
    return (C_at_satur)
    

def estimate_satur_percentage ( C , temp , altitude_m=10 ):
    #=======inputs:======
    # C: DO concentration 
    # temp: temperature value at C concentration 
    # altitude_m: Lake altitude in m (for lake Erken it is 10 m )
    #=======outputs:=======
    #est_satur: estimated oxygen saturation level 
    
    est_satur=(C/C_satur (temp , 1))*100
        
    return (est_satur)

#%% Calculating the DO values at the boundary layer (Z_m= 13.5m from surface) during stratified period 


def daily_C_bnd( temp_TB_str, temp_TB_initial,  delta_t, initial_cond_sat, bnd_layer_index= 25,  jz_bnd=0.366): 
    #temp_TB_str: Temperature full profile during stratified period 
    #temp_TB_initial: Temperature full profile on the day before stratification onset where the initial concentration has been identified 
    #delta_t: Time difference between each two stratified period 
    #initial_cond_sat: O2 saturation percentage on the day before onset in float number ranged between 0 and 1 (e.g. initial_cond_sat=0.8 means 80% saturation)
    #bnd_layer_index: The index of the boundary layer where the surface layer index is 0 in Erken the boundary layer (Z_m= 13.5m) index is 25
    #jz_bnd: The estimated oxygen depletion rate at the boundary layer 

    temp_daily_str_bnd= temp_TB_str[: , bnd_layer_index] # temperature at the boundary layer during stratified period
    temp_bnd_inital_cond= temp_TB_initial[bnd_layer_index] # initial (a day before onset) boundary layer temperature 

    
    m = len(delta_t) # number of stratified days
    C = np.zeros( m+1 ) # add one 0 at the end of stratified period DO at the boundary layer
    
    
    

    C_initial_bnd_cond = C_satur(temp_bnd_inital_cond, initial_cond_sat)# calculate initial saturated DO level at boundary layer
    C[0] = C_initial_bnd_cond
    
    for i in range(0, m):
        
        
        C[i+1]= C[i] - jz_bnd*(K_temp(temp_daily_str_bnd.mean())) # Jz at boundary layer is modified by temperature during stratified period

    
    C = np.where(C < 0, 0, C)

    C_str_bnd = C[1:] # removing the initial condition concentration to have oxygen at boundary layer in the stratified period 
    
    return ( C_str_bnd )







  

#%%DO model simulation for one stratified period 

#=======inputs are:==============
#1) params: a list of model parameters including jv (oxygen consumption rate from water-column)  
### and ja (oxygen consumption rate from sediment)
###   (the time step for solving the differential equations in unit of day which in Erken was 1 [d])
#2) hypo_prof_morpho: a list of  morphometrical variables including 
### volume, surface area, lateral area of each layer in deepwater and the layer thickness (in Erken 0.5) 
#3) temp_hypo_2D_str: A 2-D temperature array in deepwater layers (in columns)  during stratified period (each row one day)
#4) temp_TB_2D_str: A 2-D array of Top to Bottom (TB) temperature profile during stratified period 
#5) temp_TB_initial: A 2-D array of Top to Bottom (TB) temperature profile on a day before stratification starts 
#6) initial_cond_sat: Oxygen saturation level on initial day before the stratification starts 
#7) duration_repl: number of replenishment days (for the first stratification duration in year it is 0) 
#8) delta_t: An array of time difference between each two stratified days
#9) kz_hypo_str_2D_daily: A 2-D array of daily deepwater mixing coefficient (kz) profile values during stratified period 


#=====Outputs are:==============
#1) C_str_daily_model: a 2-D array of DO deepwater daily profile
#2) sat_turnover:  The average of DO saturation level in deepwater on turnover date 


def simulate_deepwater_DO_one_str(params, hypo_prof_morpho, temp_hypo_2D_str, temp_TB_2D_str, temp_TB_initial , 
                                  initial_cond_sat, duration_repl, delta_t, kz_hypo_str_2D_daily,  bnd_layer_index= 25,):
                        

    jv, ja = params
    V_r, A_s, A_l, dz  = hypo_prof_morpho
    nl= len(V_r)# number of layer in deepwater 


    
    # Initial DO condition identification based on the day before onset temperature and saturation level  
    temp_hypo_initial= temp_TB_initial [(bnd_layer_index +1): ]# hypolimnion or deepwater is identified deeper than the boundary layer 
    
    C_hypo_initial = C_satur(temp_hypo_initial, initial_cond_sat)
    
    
    
    m = len(delta_t)
    

        
    temp_hypo_for_jz_correction = temp_hypo_2D_str.T# now in the shape of (depth, date)
    
    
    kz_hypo_str= (24 * 3600 * kz_hypo_str_2D_daily).T# now Kz in the shape of (depth, date) and in unite of m2/d 


      
    
    C = np.zeros((nl, m+1))# C in the shape of (number of layers) x (number of stratified days +1; but later on this added day with oxygen value of 0 will be removed) 
    
    
    C[:, 0] = C_hypo_initial # setting the oxygen concentration onset at a saturated level 


    C_bnd_daily= daily_C_bnd( temp_TB_2D_str, temp_TB_initial , delta_t, initial_cond_sat ) # identification of daily DO values at boundary layer 

    
    #============Solving the daily differential equations using the impilicit approach==========
    Factor_dig = np.zeros((nl, nl)) 


    for j in range(1, m+1):# day index

        
        Factor_dig= np.diag([ 1  +delta_t[j-1]*A_s[1]*kz_hypo_str[1, j-1]/(V_r[0]*dz)+ delta_t[j-1]*A_s[0]*kz_hypo_str[0,j-1]/(dz*V_r[0]),
                            1+ delta_t[j-1]*A_s[1]*kz_hypo_str[1,j-1]/(dz*V_r[1]) + delta_t[j-1]*A_s[2]*kz_hypo_str[2,j-1]/(dz*V_r[1]),
                            1+ delta_t[j-1]*A_s[2]*kz_hypo_str[2,j-1]/(dz*V_r[2]) + delta_t[j-1]*A_s[3]*kz_hypo_str[3,j-1]/(dz*V_r[2]), 
                            1+ delta_t[j-1]*A_s[3]*kz_hypo_str[3,j-1]/(dz*V_r[3]) + delta_t[j-1]*A_s[4]*kz_hypo_str[4,j-1]/(dz*V_r[3]),
                            1+ delta_t[j-1]*A_s[4]*kz_hypo_str[4,j-1]/(dz*V_r[4]) + delta_t[j-1]*A_s[5]*kz_hypo_str[5,j-1]/(dz*V_r[4]),
                            1+ delta_t[j-1]*A_s[5]*kz_hypo_str[5,j-1]/(dz*V_r[5]) + delta_t[j-1]*A_s[6]*kz_hypo_str[6,j-1]/(dz*V_r[5]),
                            1+ delta_t[j-1]*A_s[6]*kz_hypo_str[6,j-1]/(dz*V_r[6]) + 0 ] , 0)- \
                    np.diag([ delta_t[j-1]*A_s[1]*kz_hypo_str[1,j-1]/(V_r[0]*dz) , 
                                delta_t[j-1]*A_s[2]*kz_hypo_str[2,j-1]/(dz*V_r[1]),
                                delta_t[j-1]*A_s[3]*kz_hypo_str[3,j-1]/(dz*V_r[2]), 
                                delta_t[j-1]*A_s[4]*kz_hypo_str[4,j-1]/(dz*V_r[3]),
                                delta_t[j-1]*A_s[5]*kz_hypo_str[5,j-1]/(dz*V_r[4]),
                                delta_t[j-1]*A_s[6]*kz_hypo_str[6,j-1]/(dz*V_r[5])] , 1)- \
                        np.diag([ delta_t[j-1]*A_s[1]*kz_hypo_str[1,j-1]/(dz*V_r[1]),
                                    delta_t[j-1]*A_s[2]*kz_hypo_str[2,j-1]/(dz*V_r[2]), 
                                    delta_t[j-1]*A_s[3]*kz_hypo_str[3,j-1]/(dz*V_r[3]),
                                    delta_t[j-1]*A_s[4]*kz_hypo_str[4,j-1]/(dz*V_r[4]),
                                    delta_t[j-1]*A_s[5]*kz_hypo_str[5,j-1]/(dz*V_r[5]),
                                    delta_t[j-1]*A_s[6]*kz_hypo_str[6,j-1]/(dz*V_r[6]) ] , -1)
                        
    
                        
                        
            
                        
        b = C[:, j-1].copy()

        b[0] = b[0]+  ((C_bnd_daily [j-1]*kz_hypo_str[0,j-1]* delta_t[j-1] * A_s[0])/(dz * V_r[0]))
        


        knows=b - (delta_t[j-1] * K_temp(temp_hypo_for_jz_correction[:, j-1])* (  jv + (A_l / V_r) * ja  ))

        
        solution = np.linalg.solve(Factor_dig, knows)
       
        
        
        C[:, j] = solution
       
        C = np.where(C < 0, 0, C)# replacing the negetive values of DO with 0
    #================================================================================================
    
    
    C = C.T # Reshaping DO array into the size of (day x layer)

    C_str = C[1:, :] # removing the initial condition day oxygen value

    C_str_ave_model = np.dot(C_str, V_r) / sum(V_r) # calculating the deepwater daily average of oxygen 
    
    #calculating saturation turnover value 
    DO_ave_turnover = C_str_ave_model[-1] # the turnover date oxygen average value 
    temp_ave_turnover = np.dot(temp_hypo_2D_str[-1, :], V_r) / sum(V_r) # the turnover date temperature average value 
    sat_turnover = DO_ave_turnover / C_satur(temp_ave_turnover, 1) # Oxygen saturation level on turnover date 

   

    return C_str, sat_turnover 


#%% Function to calculate deepwater DO profiles during stratification
# Based on daily temperature and vertical diffusivity (Kz) profiles.
# NOTE: The function returns a DO array matching the shape of the daily 
# deepwater temperature array. Non-stratified profiles are marked with NaN.     

#%% Note on the input data for the oxygen function:


def simulate_deepwater_DO (params, temp_TB_2D, kz_TB_2D, dens_diff_bnd_surf,
                               hypo_prof_morpho= hypo_prof_morpho_Erken, bnd_layer_index= 25, 
                               replenishment_rate= 2.7946):
    
    #1) params: The two model parameters including jv (water-column oxygen consumption rate), and ja (sediment oxygen consumption rate)
    #2) temp_TB_2D:  A 2-D array of Top to Bottom temperature daily profiles (based on all temperature recodes not only stratified periods)
    #3) kz_TB_2D: A 2-D array of Top to Bottom Kz daily profiles (has the same size as temp_TB_2D)
    
    #=Note: Although it may seem unusual to use "kz_TB_2D" (full-depth Kz) instead of 
    # "kz_deepwater_2D", this approach is intentional. It provides a flexible 
    # framework that is compatible with more advanced diffusivity calculations 
    # (e.g., using a hydrodynamic model like GOTM), and prepares the structure 
    # for future climate projections.
    
    #4) dens_diff_bnd_surf: daily density difference between the boundary layer and surface water 
    #5) hypo_prof_morpho: a list of  morphometrical variables including 
    ### volume, surface area, lateral area of each layer in deepwater and the layer thickness (in Erken 0.5) 
    #6) bnd_layer_index: index of boundary layer in Erken (25; Z_m+= 13.5m )
    #7) replenishment_rate: the estimated replenishment rate in one year with mid-summer mixing in Erken 2.7946 %/d
    
    temp_deepwater_2D= temp_TB_2D[: , bnd_layer_index+1: ] # seperating deepwater (below boundary layer) temperature profile (Z_m< 13.5) 
    kz_deepwater_2D= kz_TB_2D[: , bnd_layer_index+1: ]# seperating deepwater (below boundary layer) temperature profile (Z_m< 13.5) 
    
    
    temp_surf= temp_TB_2D[: , 0]# all dates surface temperature 
       
    
    df_str_periods_info= find_str_period ( dens_diff_bnd_surf , temp_surf ) # Identify stratified periods 
    
    
    
    DO_2D_deepwater_model = np.full(temp_deepwater_2D.shape, np.nan) # Identify the oxygen 2-D profile in the same size of temperature profile with nan values 
    
    
    
    # Based on information in "df_str_periods_info" the oxygen profile will be calculated
    # and be placed in "DO_2D_deepwater_model"
    
    for index, row in df_str_periods_info.iterrows():
        
        #==== Preperation all the input data for using the "simulate_deepwater_DO_one_str" function ===============
        index_str_period_in_year=row['index_str_period_in_year'] # index of esch stratified period 
        
        # subseting the deepwater temperature, Top to Bottom temperature, initial condition temperature    
        # and Kz profiles for stratified periods (+1 is added to make sure the turnover day will be also included)
        temp_hypo_str= temp_deepwater_2D[int(row['onset_indices']) : int(row['end_indices'])+1] 
        temp_TB_str=  temp_TB_2D[int(row['onset_indices']) : int(row['end_indices'])+1]
        temp_TB_inital_cond= temp_TB_2D[int(row['initial_cond_indices'])]
        kz_hypo_str=  kz_deepwater_2D[int(row['onset_indices']) : int(row['end_indices'])+1]
        
        
        duration_repl= row['replenishment_duration']# replenishment period duration 
        str_duration= row['str_duration'] # number of stratified period 
        delta_t=np.ones(str_duration)# the time step between temperature profile records [d] in Erken was 1 [d]

        # if it is the first stratified period in a year the DO saturation level is 100% at the initial condition temperature   
        if index_str_period_in_year==0:
            initial_cond_sat=1 # a float number for 100% saturation condition 
            C_str_daily_model, sat_end = simulate_deepwater_DO_one_str(params, hypo_prof_morpho, temp_hypo_str,
                                                             temp_TB_str, temp_TB_inital_cond, initial_cond_sat, duration_repl,
                                                             delta_t, kz_hypo_str ) 
                                

        # otherwise the saturation level on the initial condition date will be calculated from 
        # the previous turnover saturation level, the replenishment duration and 
        # the estimated "replenishment_rate [%/d]" 
        else:
            
            
            increase_sat_repl = duration_repl * (replenishment_rate / 100)
        
            initial_cond_sat = sat_end + increase_sat_repl

            C_str_daily_model, sat_end  = simulate_deepwater_DO_one_str(params,hypo_prof_morpho, temp_hypo_str,
                                                              temp_TB_str, temp_TB_inital_cond, initial_cond_sat, duration_repl,
                                                              delta_t, kz_hypo_str)
                       

            
        # replacing the nan values over the stratified period with model simulations 
        DO_2D_deepwater_model[int(row['onset_indices']) : int(row['end_indices'])+1 ]= C_str_daily_model   

    
    return DO_2D_deepwater_model 


#%% Making ready all the input data for the oxygen simulation function 

#%% Creating kz_TB_2D from Kz calculated from heat budget in the stratified period

# The heat budget-based calculations are only valid during stratified periods. 
# Therefore, Kz values were adapted based on the stratification duration, 
# and filled with 0 for non-stratified days.
# However, when calculating deepwater DO profiles, only the stratified 
# deepwater Kz values were used. (see "kz_deepwater_2D" in definition of "simulate_deepwater_DO" function)


df_bas= temp_daily_prof_Erken_2019_2022.reset_index()
df_sub=pd.concat( [ df_info_prof_str_2019_0, df_info_prof_str_2019_1 , df_info_prof_str_2020_0, df_info_prof_str_2021_0, df_info_prof_str_2022_0]).reset_index()



merged_df= df_bas.merge(df_sub[['Datetime', 'Z_m+', 'kz']], left_on=['Datetime', 'Z_m+'], right_on=['Datetime', 'Z_m+'], how='left')


# Fill NaN values with 0 in 'kz' (for non-stratified days or surface Kz)

merged_df['kz'].fillna(0, inplace=True)


#Generating a 2-D array from Kz column 
kz_TB_2D_2019_2022 = two_D_array_from_df(merged_df, values_name='kz', index_name='Datetime')  
    

#%% Generating a 2-D array of temperature profile 

#a 2-D array from temp column in daily profile dataframe
temp_TB_2D_2019_2022=  two_D_array_from_df (temp_daily_prof_Erken_2019_2022, 'Temp', index_name='Datetime') 




#%% Estimating the two model parameters based on the Root Mean Square Error (RMSE) in 
# simulating the deepwater DO profile over 2020 and 2021 in Lake Erken 

from sklearn.metrics import mean_squared_error
# pip install hydroeval if not already installed
from hydroeval import evaluator, nse


# Function to calculate model error metrics (RMSE and NSE) for simulated deepwater DO profiles
def calc_deepwater_DO_profile_errors(params):
    # Unpack parameter values (vertical and areal oxygen flux parameters)
    jv, ja = params

    # Extract unique time indices from density difference data
    time_series = density_diff_Erken_2019_2022.index.unique()

    # Simulate deepwater DO profiles using the input model
    DO_deepwater_prof_model_2D_2019_2022 = simulate_deepwater_DO(
        params,  # tuple (jv, ja)
        temp_TB_2D_2019_2022,  # 2D temperature profile (full water column)
        kz_TB_2D_2019_2022,  # 2D vertical diffusivity profile (full water column)
        density_diff_Erken_2019_2022,  # stratification strength indicator
        hypo_prof_morpho_Erken,  # morphological information of the hypolimnion
        bnd_layer_index=25,  # index for boundary layer (top of hypolimnion)
        replenishment_rate=2.7946  # rate of reoxygenation (e.g., from inflow or surface mixing)
    )

    # Convert model output to a DataFrame indexed by time
    DO_deepwater_prof_model_2D_2019_2022_indexed = pd.DataFrame(
        DO_deepwater_prof_model_2D_2019_2022, index=time_series
    )

    # Filter to only include data from years 2020 and 2021, dropping any rows with NaNs (stratified period will only have remained)
    DO_deepwater_prof_model_2D_2020_2021_indexed = DO_deepwater_prof_model_2D_2019_2022_indexed[
        (DO_deepwater_prof_model_2D_2019_2022_indexed.index.year == 2020) |
        (DO_deepwater_prof_model_2D_2019_2022_indexed.index.year == 2021)
    ].dropna()

    # Convert to NumPy array for metric calculation
    DO_deepwater_prof_model_2D_2020_2021 = np.array(DO_deepwater_prof_model_2D_2020_2021_indexed)

    # === Observation Data Preparation ===

    # Combine stratified profiles from 2020 and 2021
    df_str_2020_2021 = pd.concat([df_info_prof_str_2020_0, df_info_prof_str_2021_0])

    # Filter for deepwater layers (e.g., Z > 13.5 m)
    df_deepwater_str_2020_2021 = df_str_2020_2021[df_str_2020_2021['Z_m+'] > 13.5]

    # Convert observations to 2D array: rows = dates, columns = depths
    DO_deepwater_prof_obs_2D_2020_2021 = two_D_array_from_df(
        df_deepwater_str_2020_2021, values_name='DO', index_name='Datetime')
    

    # === Evaluation Metrics ===

    # Mean Squared Error (MSE)
    MSE_prof = mean_squared_error(
        DO_deepwater_prof_model_2D_2020_2021,
        DO_deepwater_prof_obs_2D_2020_2021)
    

    # Nash-Sutcliffe Efficiency (NSE)
    # Flatten 2D arrays into 1D for metric calculation
    NSE_prof = evaluator( nse,
        DO_deepwater_prof_model_2D_2020_2021.flatten(),
        DO_deepwater_prof_obs_2D_2020_2021.flatten())
    

    # Root Mean Squared Error (RMSE)
    RMSE_prof = math.sqrt(MSE_prof)

    # Return both RMSE and NSE as performance indicators 
    # The NSE values were used as an estimate of the likelihood function 
    return (RMSE_prof, NSE_prof)

#%% Testing different parameters and saving the model performance in a dataframe "df_calib_2020_2021" 

import math
import matplotlib.pyplot as plt

# Define the parameter ranges and number of points
param1_name = 'J$_{V}$ [g m$^{-3}$ d$^{-1}$]'
param2_name = 'J$_{A}$ [g m$^{-2}$ d$^{-1}$]'


# maximum values are selected based on the previous studies in mesotrophic lakes  

jv_min, jv_max = 0.001, 0.84
ja_min, ja_max = 0.001, 1.7 

# The testing values of jv and ja were selected from grid sampling with uniform step sizes of 0.1 
num_points1 =len(np.arange(jv_min, jv_max+0.1 , 0.1))
num_points2 =len(np.arange(ja_min , ja_max+0.1 , 0.1))

param1_values = np.arange(jv_min, jv_max+0.1 , 0.1)
param2_values = np.arange(ja_min , ja_max+0.1 , 0.1)

#Determine two arrays for the results of performance saving 
rmse_calib_prof = np.zeros((num_points1, num_points2))
nse_calib_prof = np.zeros((num_points1, num_points2))





data_perfo_test_2020_2021 = {'Jv': [], 'Ja': [],
                        'rmse_prof': [],
                        'nse_prof': []}

for i, p1 in enumerate(param1_values):
    for j, p2 in enumerate(param2_values):

        rmse_calib_prof[i, j]  = calc_deepwater_DO_profile_errors([p1, p2])[0]
        nse_calib_prof[i, j]  = calc_deepwater_DO_profile_errors([p1, p2])[1]
        
        # Append the values to the data lists
        data_perfo_test_2020_2021['Jv'].append(p1)
        data_perfo_test_2020_2021['Ja'].append(p2)
        

        data_perfo_test_2020_2021['rmse_prof'].append(rmse_calib_prof[i, j])
        data_perfo_test_2020_2021['nse_prof'].append(nse_calib_prof[i, j])
        

               
# Create a dataframe from the data
df_perfo_test_2020_2021 = pd.DataFrame(data_perfo_test_2020_2021)




#%% Create a contour plot to visualize model performance in estimating deepwater oxygen profiles (RMSE)



# Create a meshgrid of parameter values (jv on Y-axis, ja on X-axis)
param2_mesh, param1_mesh = np.meshgrid(param2_values, param1_values)  # Meshgrid order: (columns, rows)

# Set global plot font style and size
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 22

# Create the figure and set size
plt.figure(figsize=(12, 10))

# Round down the minimum RMSE to one decimal for better axis alignment
min_rmse = df_perfo_test_2020_2021.rmse_prof.min()
min_rmse_range = math.floor(min_rmse * 10) / 10

# Round up the maximum RMSE to one decimal for better axis alignment
max_rmse = df_perfo_test_2020_2021.rmse_prof.max()
max_rmse_range = math.ceil(max_rmse * 10) / 10

# Create custom contour levels at 0.1 intervals
level_custom = np.arange(min_rmse_range, max_rmse_range + 0.15, 0.15)


# Create filled contour plot for RMSE values across parameter space
contourf_plot = plt.contourf(param1_mesh, param2_mesh, rmse_calib_prof, levels=level_custom, cmap='RdYlGn_r')
plt.colorbar(contourf_plot, label='Deepwater DO profile RMSE [mg L$^{-1}$]')

# Add contour lines with RMSE values
contour_lines = plt.contour(param1_mesh, param2_mesh, rmse_calib_prof, levels=level_custom, colors='black')
plt.clabel(contour_lines, inline=True, fontsize=21)

# Set axis labels
plt.xlabel('J$_{V}$ [g m$^{-3}$ d$^{-1}$]', fontsize=22)
plt.ylabel('J$_{A}$ [g m$^{-2}$ d$^{-1}$]', fontsize=22)

# Define tick intervals on axes
plt.xticks(np.arange(0.1, 0.85, 0.1))
plt.yticks(np.arange(0.2, 1.7, 0.2))

# Adjust tick appearance
plt.tick_params(axis='both', which='both', width=2, length=10)

# Save the figure 
plt.savefig(output_dir / "testing_parameters_rmse_deepwater_prof_2020_2021.png", dpi=300)
plt.show()


#%% Finding the acceptable parameter sets with a threshold of 0.8 mg/L in RMSE profile: 
    

rmse_prof_threshold= 0.75

df_acceptable_params= df_perfo_test_2020_2021[df_perfo_test_2020_2021['rmse_prof']<rmse_prof_threshold].reset_index()

len(df_acceptable_params)# 30 parameter sets deemed acceptable and were used for further projection

#%% Adding a column of "VHDO" (Hypolimnetic Volumetric Oxygen Demand) 

morpho_ratio_deepwater = sum(hypo_prof_morpho_Erken[2])/sum(hypo_prof_morpho_Erken[0]) # 0.4851 # morphological ratio based on Livingstone and Imboden 1996 
#is the ratio of sediment area to volume of deepwater 


df_acceptable_params['VHOD']= df_acceptable_params['Jv']+ df_acceptable_params['Ja']* morpho_ratio_deepwater


df_acceptable_params['VHOD'].median()#0.468 mg L-1 d-1


# === Save to CSV in output directory ====
df_acceptable_params.to_csv(output_dir / 'df_acceptable_params.csv', index=False)

#%% Run the deepwater DO model using all accepted parameter combinations 
# and save the output results to separate CSV files in the specified output directory


os.chdir(output_dir)



for index, row in df_acceptable_params.iterrows():
    p1 = row['Jv']
    p2 = row['Ja']
    p3 = row['nse_prof']
    
    header_values = {'Jv': p1, 'Ja': p2 , 'nse_prof': p3 }  
    header_df = pd.DataFrame([header_values])
    
    # Round Ja and Jv to 3 decimal places
    Jv_name = round(p1, 3)
    Ja_name = round(p2, 3)

    result_do = simulate_deepwater_DO ([p1, p2],  temp_TB_2D_2019_2022,  kz_TB_2D_2019_2022, density_diff_Erken_2019_2022,
                               hypo_prof_morpho= hypo_prof_morpho_Erken, bnd_layer_index= 25, 
                               replenishment_rate= 2.7946)
    

    # Create a DataFrame for the result
    result_do = pd.DataFrame(result_do, columns=[f'{depth}m' for depth in np.arange(-14 , -17.5 , -0.5)])

    # Add a column for dates
    result_do.insert(0, 'Datetime', density_diff_Erken_2019_2022.index)  # Replace 'your_date_array' with your actual date array

    # Define the filename_do
    filename_do = f'DO_deepwater_prof_2019_2022_Jv_{Jv_name}_Ja_{Ja_name}.csv'

    # add jv, ja and weights values in the first row 
    header_df = pd.DataFrame([header_values])
    header_df.to_csv(filename_do, index=False)


    # Save the result to a CSV file
    do_result = pd.DataFrame(result_do)  # Create a DataFrame for result
    do_result.to_csv(filename_do, mode='a', index=False, na_rep='NaN')  


#%%Estimating the 90% confidence band in model simulations using the  
#"weighted_quantiles" function. The likelihood_value in these functions are acceptable parameters NSE values. 

 
def weighted_quantiles(values, quantiles, likelihood_values=None):
    """ Modified from 
        http://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy     
        NOTE: quantiles should be in [0, 1]
        
        values         array with data
        quantiles      array with desired quantiles
        likelihood_value  array of weights (the same length as `values`)

        Returns array with computed quantiles.
    """
    # Convert to arrays
    values = np.array(values)
    quantiles = np.array(quantiles)
    
    
    # Assign equal weights if necessary
    if likelihood_values is None:
        likelihood_values = np.ones(len(values))
        
    # Otherwise use specified weights
    likelihood_values = np.array(likelihood_values)
    
    # Check quantiles specified OK
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), 'quantiles should be in [0, 1]'

    # Sort 
    sorter = np.argsort(values)
    values = values[sorter]
    likelihood_values = likelihood_values[sorter]

    # Compute weighted quantiles
    weighted_quantiles = (np.cumsum(likelihood_values) - 0.5 * likelihood_values)
    weighted_quantiles /= np.sum(likelihood_values)
    
    return np.interp(quantiles, weighted_quantiles, values) 


#%% Calculating hypolimnetic volume-weighted average in deepwater

def hypolimnetic_ave(C, Vr, time_series):
    C_ave=np.dot (C , Vr) / sum (Vr)
    C_ave_indexed=pd.Series(C_ave, index=pd.to_datetime(time_series))
    return (C_ave_indexed) 

#%%Aggregating all the acceptable simulations of deepwater DO in one dataframe (allocating one column for each jv and ja combination) 

all_deepwater_DO_ave_model=  pd.DataFrame([])

# Loop through each CSV file in the directory
for filename in os.listdir(output_dir):
    if filename.endswith(".csv") and filename.startswith("DO_deepwater_prof"):
        
        # Read from CSV
        read_header_df = pd.read_csv(os.path.join(output_dir, filename), nrows=1)
        jv_value = read_header_df['Jv'].iloc[0]
        ja_value = read_header_df['Ja'].iloc[0]
        nse_prof = read_header_df['nse_prof'].iloc[0]
        
        # Read the DataFrame below the header
        read_data_df = pd.read_csv(os.path.join(output_dir, filename), skiprows=2)
        # Extract relevant columns (adjust column names accordingly)
        C = read_data_df.iloc[:, 1:].values  # Assuming your concentrations are in columns 1 and onwards

        time_series = pd.to_datetime(read_data_df['Datetime'])
        
        deepwater_DO_ave_model = hypolimnetic_ave(C, hypo_prof_morpho_Erken[0] , time_series)
        
        all_deepwater_DO_ave_model = pd.concat([all_deepwater_DO_ave_model, deepwater_DO_ave_model], axis=1)

#%% 90% confidence band calculation for average daily deepwater DO based on 
# General likelihood Uncertainty Estimation (GLUE) approach and likelihood of acceptable parameters NSEs 

 
quants = [0.05 ,0.5 , 0.95 ]  
out=[]    
for index, row in all_deepwater_DO_ave_model.iterrows():# for each timesteps
    values = row    
    out.append(weighted_quantiles(values, quants, likelihood_values =df_acceptable_params['nse_prof']))

    # Build df
    glue_deepwater_DO_average = pd.DataFrame(data=out, columns=['5%', '50%', '95%'])

glue_deepwater_DO_average= glue_deepwater_DO_average.set_index(time_series)



#%% Subsetting the daily hypolimnetic DO average estimations for each year

glue_deepwater_ave_model_2019= glue_deepwater_DO_average[glue_deepwater_DO_average.index.year==2019]

glue_deepwater_ave_model_2020= glue_deepwater_DO_average[glue_deepwater_DO_average.index.year==2020]

glue_deepwater_ave_model_2021= glue_deepwater_DO_average[glue_deepwater_DO_average.index.year==2021]

glue_deepwater_ave_model_2022= glue_deepwater_DO_average[glue_deepwater_DO_average.index.year==2022]



#%% Calculating daily hypolimnetic DO average in observational records  

#2019_0
df_deepwater_str_2019_0= df_info_prof_str_2019_0[df_info_prof_str_2019_0['Z_m+'] > 13.5]
deepwater_obs_DO_ave_2019_0 = hypolimnetic_ave(two_D_array_from_df(df_deepwater_str_2019_0, values_name='DO', index_name='Datetime'), hypo_prof_morpho_Erken[0] , df_deepwater_str_2019_0.index.unique() )

#2019_1
df_deepwater_str_2019_1= df_info_prof_str_2019_1[df_info_prof_str_2019_1['Z_m+'] > 13.5]
deepwater_obs_DO_ave_2019_1= hypolimnetic_ave(two_D_array_from_df(df_deepwater_str_2019_1, values_name='DO', index_name='Datetime'), hypo_prof_morpho_Erken[0] , df_deepwater_str_2019_1.index.unique() )

#Aggregating two stratified period results in 2019 dataframe 
deepwater_obs_DO_ave_2019=pd.concat([deepwater_obs_DO_ave_2019_0, deepwater_obs_DO_ave_2019_1])

#2020_0
df_deepwater_str_2020_0= df_info_prof_str_2020_0[df_info_prof_str_2020_0['Z_m+'] > 13.5]
deepwater_obs_DO_ave_2020= hypolimnetic_ave(two_D_array_from_df(df_deepwater_str_2020_0, values_name='DO', index_name='Datetime'), hypo_prof_morpho_Erken[0] , df_deepwater_str_2020_0.index.unique() )

#2021_0
df_deepwater_str_2021_0= df_info_prof_str_2021_0[df_info_prof_str_2021_0['Z_m+'] > 13.5]
deepwater_obs_DO_ave_2021= hypolimnetic_ave(two_D_array_from_df(df_deepwater_str_2021_0, values_name='DO', index_name='Datetime'), hypo_prof_morpho_Erken[0] , df_deepwater_str_2021_0.index.unique() )

#2022_0
df_deepwater_str_2022_0= df_info_prof_str_2022_0[df_info_prof_str_2022_0['Z_m+'] > 13.5]
deepwater_obs_DO_ave_2022= hypolimnetic_ave(two_D_array_from_df(df_deepwater_str_2022_0, values_name='DO', index_name='Datetime'), hypo_prof_morpho_Erken[0] , df_deepwater_str_2022_0.index.unique() )



#%%Creating a plot of model performance in simulating daily deepwater DO from 2019 to 2022 

fig, axes = plt.subplots(2, 2, figsize=(22, 23))

for ax in axes.flat:  
    ax.grid(False)
    ax.tick_params(bottom=True,  left=True)
    ax.tick_params(axis='both', which='both', width=2, length=10)

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)
        
# Plot for 2020 calibration period
axes[0, 0].fill_between(glue_deepwater_ave_model_2020.index, glue_deepwater_ave_model_2020['5%'], glue_deepwater_ave_model_2020['95%'], color='red', alpha=0.4, label='5-95% model bound')
axes[0, 0].plot(glue_deepwater_ave_model_2020.index, glue_deepwater_ave_model_2020['50%'], 'k--', label='Median model', alpha=0.9)
axes[0, 0].scatter(deepwater_obs_DO_ave_2020.index, deepwater_obs_DO_ave_2020, color='k', label='Obs')
axes[0, 0].set_ylabel('Deepwater DO average [mg L$^{-1}$]', fontsize=36)
axes[0, 0].set_ylim([0, 12.5])
axes[0, 0].tick_params(axis='x', rotation=40)
axes[0, 0].tick_params(axis='both', which='both', width=2, length=10)
axes[0, 0].text(0.98, 0.98, '(a)', transform=axes[0, 0].transAxes, fontsize=34, ha='right', va='top')

# Plot for 2021 calibration period
axes[0, 1].fill_between(glue_deepwater_ave_model_2021.index, glue_deepwater_ave_model_2021['5%'], glue_deepwater_ave_model_2021['95%'], color='red', alpha=0.4)
axes[0, 1].plot(glue_deepwater_ave_model_2021.index, glue_deepwater_ave_model_2021['50%'], 'k--', alpha=0.9)
axes[0, 1].scatter(deepwater_obs_DO_ave_2021.index, deepwater_obs_DO_ave_2021, color='k')
axes[0, 1].set_ylim([0, 12.5])
axes[0, 1].tick_params(axis='x', rotation=40)
axes[0, 1].tick_params(axis='both', which='both', width=2, length=10)
axes[0, 1].text(0.98, 0.98, '(b)', transform=axes[0, 1].transAxes, fontsize=34, ha='right', va='top')

# Plot for 2019 validation period
axes[1, 0].fill_between(glue_deepwater_ave_model_2019.index, glue_deepwater_ave_model_2019['5%'], glue_deepwater_ave_model_2019['95%'], color='red', alpha=0.4)
axes[1, 0].plot(glue_deepwater_ave_model_2019.index, glue_deepwater_ave_model_2019['50%'], 'k--', alpha=0.9)
axes[1, 0].scatter(deepwater_obs_DO_ave_2019.index, deepwater_obs_DO_ave_2019, color='k')
axes[1, 0].set_ylabel('Deepwater DO average [mg L$^{-1}$]', fontsize=36)
axes[1, 0].set_ylim([0, 12.5])
axes[1, 0].tick_params(axis='x', rotation=40)
axes[1, 0].tick_params(axis='both', which='both', width=2, length=10)
axes[1, 0].text(0.98, 0.98, '(c)', transform=axes[1, 0].transAxes, fontsize=34, ha='right', va='top')

# Plot for 2022 validation period
axes[1, 1].fill_between(glue_deepwater_ave_model_2022.index, glue_deepwater_ave_model_2022['5%'], glue_deepwater_ave_model_2022['95%'], color='red', alpha=0.4)
axes[1, 1].plot(glue_deepwater_ave_model_2022.index, glue_deepwater_ave_model_2022['50%'], 'k--', alpha=0.9)
axes[1, 1].scatter(deepwater_obs_DO_ave_2022.index, deepwater_obs_DO_ave_2022, color='k')
axes[1, 1].set_ylim([0, 12.5])
axes[1, 1].tick_params(axis='x', rotation=40)
axes[1, 1].tick_params(axis='both', which='both', width=2, length=10)
axes[1, 1].text(0.98, 0.98, '(d)', transform=axes[1, 1].transAxes, fontsize=34, ha='right', va='top')

# bigger font size in both axes' elements  
for ax in axes.flatten():
    ax.tick_params(axis='both', labelsize=36)

fig.legend(['Model 90% CI', 'Model median', 'Observations'], 
           loc='upper center', 
           bbox_to_anchor=(0.5, -0.01),  # Adjust this to move legend up/down
           fontsize=38, 
           ncol=3)

plt.tight_layout()
plt.savefig("model_performance_hypolimnetic_DO_daily_ave.png", dpi=300, bbox_inches='tight')
plt.show()
