import numpy as np
import pandas as pd
import os
from datetime import datetime
from os import listdir as ld

"""
Python script written to find periods of valid wear separated by days then calculates the total minutes of each activity level for each day

Assumes filenames are in the form 'MOSTID_*.csv'

Returns a csv with activity minutes per day for each MOST participants to current working directory

Created by Peter Schaefer at the University of Florida AI Biomechanics Laboratory
"""

# progress log
print('Packages imported\nScript running',flush=True)

# Path to physical activity and wear time validation files
pa_path = ''
wtv_path = ''

# create list of file paths 
pa_list = [os.path.join(pa_path,x) for x in sorted(ld(pa_path))]
wtv_list = [os.path.join(wtv_path,x) for x in sorted(ld(wtv_path))]

# Number of files to iterate through
n = len(pa_list)

# progress log
print(f'Directory set\nNow iterating through {n} available files',flush=True)



# initialize list to store dataframes
activity_minutes_total = []

# iterate through files
for pa,wtv in zip(pa_list,wtv_list):
    # mostid from file path
    mostid = wtv.split('/')[-1].split('_')[0]
    # progress log
    print(f'Loading {mostid}',flush=True)

    # load wear time validation and physical activity data
    loaded_wtv = pd.read_csv(wtv,dtype={
        'time':'datetime64[ns]',
        'accel_x':float,
        'accel_y':float,
        'accel_z':float,
        'temperature':float,
        'Not Worn':bool,
    })
    loaded_pa = pd.read_csv(pa,dtype={
        'Time':'datetime64[ns]',
        'Sedentary (mins)':int,
        'Light (mins)':int,
        'Moderate (mins)':int,
        'Vigorous (mins)':int,
    }).set_index('Time')

    # progress log
    print(f'{mostid} Loaded\nProcessing...',flush=True)

    # Find the indices of the end of each day and crop the data to the last minute of each day.
    idx = [(int(x)+1) for x in np.where(loaded_pa.index.day[:-1] != loaded_pa.index.day[1:])[0]]
    
    # If there are at least 7 days of accelerometer data
    if len(loaded_pa) > 10080:
        cropped_pa_minutes = loaded_pa.iloc[idx[0]:idx[-1]+1]
    # If there are less than 7 days of accelerometer data
    elif  len(loaded_pa) < 10080:
        cropped_pa_minutes = loaded_pa.iloc[idx[0]:]

    # Resample the data to 100hz and forward fill missing values. (Note: 10000000ns = 10ms = 100 hz)
    resampled_cropped_pa_ms = cropped_pa_minutes.resample('10000000ns').asfreq().ffill().iloc[:-1]

    # find offset to 7 days
    offset = 60480000 - len(resampled_cropped_pa_ms)

    # find time delta (duration between each data point) .. should be 10ms
    td = resampled_cropped_pa_ms.index[-1] - resampled_cropped_pa_ms.index[-2]

    # if there is an offset less than 7 days, pad data with nans to 7 days
    if offset > 0:

        # pad data with nans to 7 days while creating new time index for missing values
        resampled_cropped_pa_ms = pd.concat([resampled_cropped_pa_ms,
                                             pd.DataFrame(np.nan,
                                                          index=np.arange(resampled_cropped_pa_ms.index[-1],
                                                                          resampled_cropped_pa_ms.index[-1]+(offset*td),
                                                                          td
                                                                         ),
                                                          columns=resampled_cropped_pa_ms.columns)],
                                                          axis=0,
        )

    # if there is an offset of more than 7 days, crop to 7 days
    elif offset < 0:

        # crop data to 7 days
        resampled_cropped_pa_ms = resampled_cropped_pa_ms.iloc[:offset]


    # Mask the data to set non-wear time to Nan/Null/None.
    masked_resampled_cropped_pa_ms = resampled_cropped_pa_ms.mask(loaded_wtv['Not Worn']) 
    
    # Resample back to 1 minute intervals.
    masked_resampled_cropped_pa_minutes = masked_resampled_cropped_pa_ms.resample('1min').asfreq()

    # Sum the minutes of each activity level for each day.
    daily_activity_minutes = masked_resampled_cropped_pa_minutes.resample('1D').sum()
    number_days = len(daily_activity_minutes)

    # Sum moderate and vigorous activity minutes to create moderate-vigorous activity minutes 
    mvpa  = daily_activity_minutes[['Moderate (mins)','Vigorous (mins)']].sum(min_count=1,axis=1).rename('Moderate-Vigorous (mins)')
    # daily_activity_minutes.drop(columns=['Moderate (mins)','Vigorous (mins)'],inplace=True) # uncomment to drop moderate and vigorous columns
    daily_activity_minutes = pd.concat([daily_activity_minutes,mvpa],axis=1)

    # Create series of mostid with length of total days
    mostid_series = pd.Series([mostid]*number_days,index=(daily_activity_minutes.index[0:number_days]),name='mostid')

    # Find days of valid wear (>10 hours or >600 minnutes of wear time)
    valid_days = (daily_activity_minutes.sum(axis=1)>=600).rename('Valid Day')

    # Concatenate activity minutes and valid days
    physical_activity_df = pd.concat([mostid_series,daily_activity_minutes,valid_days],axis=1)
    physical_activity_df.index.name = 'Day'
    activity_minutes_total.append(physical_activity_df)

    # progress log
    print(f'{mostid} Processed',flush=True)

pd.concat(activity_minutes_total).to_csv(os.path.join(os.getcwd(),'MOST3_pa-valid-days.csv'))