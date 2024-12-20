import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime
from os import listdir as ld
import math as m

"""
Python script written to determine mean daily minutes and % total in each activity classification 

Returns a CSV with valid days, minutes and % time in sedentary, light, and moderate-vigorous physical activity

Created by Peter Schaefer at the University of Florida AI Biomechanics Laboratory

Visit us at https://costello.mae.ufl.edu/
"""


# load dataframe
pa_metrics = pd.read_csv('<insert path to csv generated from detach-activity-metrics.py>')

# mask with valid day and drop valid day column
# set mostid as index
pa_metrics.set_index('mostid',inplace=True)
# groupby ids and sum to find number of valid days
num_valid_days = pa_metrics.groupby('mostid').sum()
valid_days = num_valid_days.drop(num_valid_days.columns[:-1],axis=1)
# drop nonvalid days
pa_metrics.mask(pa_metrics['Valid Day']==False,inplace=True)
# remove valid day column
pa_metrics.drop('Valid Day',axis=1,inplace=True)

# determine mean daily activity minutes 
daily_mean = pa_metrics.groupby('mostid').mean(numeric_only=True)
# determine percentage of each activity metric
percent_vals = daily_mean.apply(lambda x: 100*(x/daily_mean[daily_mean.columns[:4]].groupby('mostid').mean(numeric_only=True).sum(axis=1,min_count=1).values))
# rename columns 
percent_vals.columns = [f'% {x}' for x in percent_vals.columns]

# add moderate-vigorous minutes column
daily_mean = pd.concat([daily_mean,daily_mean[daily_mean.columns[2:]].sum(axis=1).rename('Moderate-Vigorous (mins)')],axis=1)

# merge activity minutes and % activity minutes
pa_metric_save = pd.concat([valid_days,daily_mean,percent_vals],axis=1).reset_index()
pa_metric_save.columns = ['MOSTID']+pa_metric_save.columns[1:].to_list()
pa_metric_save.set_index('MOSTID',inplace=True)

# sort by mostid and save as csv
pa_metric_save.reset_index().sort_values('MOSTID').to_csv()