import numpy as np
import pandas as pd
import os
import sys
from os import listdir as ld
import multiprocessing as mp 
import matplotlib.pyplot as plt
sys.path.append('/blue/k.costello-most/python-packages/')
sys.path.append('/home/schaeferp/most-ml/simple_classification/scripts/python/packages')
from nimbaldetach import nimbaldetach
import openmovement
from openmovement.load import CwaData

"""
Python module built to process AX3 .cwa files from MOST3

Assumes filenames are in the form 'MOSTID_*.cwa'

Returns processed files to current working directory

Contains modified code from Openmovement (https://github.com/openmovementproject/openmovement-python) and Nimbaldetach (https://github.com/nimbal/nimbaldetach)

Created by Peter Schaefer at the University of Florida AI Biomechanics Laboratory

Visit us at https://costello.mae.ufl.edu/
"""


class processAX3:
    def __init__(self):
        pass

    def process_file(self,filename,processed_files,WTV):

        # check if file has already been processed
        if f'{filename.split("_")[0]}_processed-wtv.csv' in processed_files and WTV == True:
            print(f'{filename} has already been processed',flush=True)
            pass
        elif f'{filename.split("_")[0]}_processed.csv' in processed_files and WTV == False:
            print(f'{filename} has already been processed',flush=True)
            pass

        else:
            # extract mostid from file name
            nm = filename.split('_')[0]

            # run openmovement dataloader
            with CwaData(filename, include_gyro=False, include_temperature=True) as cwa_data:
                #print statement to track progress
                print(f'Processing {filename}',flush=True)

                # As a pandas DataFrame
                samples = cwa_data.get_samples()

                # obtain the index when the day changes
                idx = samples[samples['time'].dt.day != samples['time'].dt.day.shift(1)].index

                # drop the first day, set index to time, resample to 10ms, backfill in missing values
                smp_crop = samples[idx[1]:].set_index('time').resample(rule='10ms',origin='start').mean().bfill()


            # get length of data
            ln = len(smp_crop)

            # determine offset from 7 days of data at 100hz (60,480,000 data points)
            diff_btw = 60480000 - len(smp_crop)

            # if data is longer, crop to 7 days, if data is shorter pad to 7 days with nans
            if diff_btw < 0:
                # crop data to 7 days
                smp_crop = smp_crop.iloc[:diff_btw]

                # set variables for wear time validation, no indexing is needed as data was longer than 7 days
                x_values = smp_crop['accel_x'].to_numpy()
                y_values = smp_crop['accel_y'].to_numpy()
                z_values = smp_crop['accel_z'].to_numpy()
                temperature_values = smp_crop['temperature'].to_numpy()

            elif diff_btw > 0:
                # find time delta (duration between each data point) .. should be 10ms
                td = smp_crop.index[-1] - smp_crop.index[-2]

                # pad data with nans to 7 days while creating new time index for missing values
                smp_crop = pd.concat([smp_crop,pd.DataFrame(np.nan, index=np.arange(smp_crop.index[-1],smp_crop.index[-1]+(diff_btw*td),td), columns=smp_crop.columns)],axis=0)


                # set variables for wear time validation, must be cropped as padded nans break wtv (100% wear time)
                x_values = smp_crop['accel_x'].iloc[:ln].to_numpy()
                y_values = smp_crop['accel_y'].iloc[:ln].to_numpy()
                z_values = smp_crop['accel_z'].iloc[:ln].to_numpy()
                temperature_values = smp_crop['temperature'].iloc[:ln].to_numpy()

            if WTV == True:

                # Define Frequencies
                accel_freq = 100
                temperature_freq = 100

                # print statement to track progress
                print(f'Calculating wear time for {filename}',flush=True)

                # Calculate Non-wear
                start_stop_df, nonwear_array = nimbaldetach(x_values=x_values, y_values=y_values, z_values=z_values,
                                                            temperature_values=temperature_values, accel_freq=accel_freq,
                                                            temperature_freq=temperature_freq)

                # Analysis
                total_wear_time = np.sum(nonwear_array)
                pct_worn = total_wear_time / len(nonwear_array) * 100


                # if the difference between 7 days and the length of the data is negative, the nonwear_array does not needed to be padded with nonwear values
                if diff_btw < 0:
                    pass
                elif diff_btw > 0:
                    # creates an array of True values to pad the nonwear_array
                    nan_arr = np.full((diff_btw),True)

                    # concatenate the nonwear_array with the nan_arr
                    nonwear_array = np.concatenate([nonwear_array,nan_arr],axis=0)

                # create series of nonwear_array with index of smp_crop
                nonwear_srs = pd.Series(nonwear_array,index=smp_crop.index,name='Not Worn')

                # save smp_crop (truncated and resampled data) and nonwear_srs (masking array) to csv
                save_df = pd.concat([smp_crop,nonwear_srs],axis=1)
                save_df.to_csv(os.path.join(os.getcwd(),f'{nm}_processed-wtv.csv'))
            
            else:
                # save smp_crop (truncated and resampled data) 
                smp_crop.to_csv(os.path.join(os.getcwd(),f'{nm}_processed-wtv.csv'))

            # print statement to track progress
            print(f'{filename} has been processed',flush=True)

            pass
