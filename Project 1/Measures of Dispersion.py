import pandas as pd
import matplotlib.pyplot as plt

# Column names to be added
column_names=['Subject_identifier','Jitter_percent','Jitter_microseconds','Jitter_rap','Jitter_ppq5','Jitter_ddp','Shimmer_percent','Shimmer_db','Shimmer_apq3','Shimmer_apq5','Shimmer_apq11','Shimmer_dda','Harmonicity_NHR_vs_HNR','Harmonicity_NHR','Harmonicity_HNR','Pitch_median','Pitch_mean','Pitch_SD','Pitch_min','Pitch_max','Pulse_no_pulses','Pulse_no_periods','Pulse_mean','Pulse_SD','Voice_fuf','Voice_no_breaks','Voice_deg_Breaks','UPDRS','PD_indicator']

#Reads data files and adds columns to data file
df = pd.read_csv('po1_data.txt', names=column_names)

#Seperates data files into seperate files if the subject has Parkinsons (PD Indicator of 1) or has no Parkinsons (PD Indicator of 0)
withpark = df[df["PD_indicator"]==1]
withoutpark = df[df["PD_indicator"]==0]

#Function to produce a  histogram for the desired data set
def cycle (a,b):
    #Print summary statistics of the "Jitter in %" column only
    print(a, "with Parkinsons:", withpark[b].describe(),)
    print(a, "without Parkinsons:",withoutpark[b].describe(),'''
          ''')


#This runs the function for each column within the data set and compares the results of each column in an overlay
cycle('Jitter in %', 'Jitter_percent')
cycle('Absolute jitter in microseconds', 'Jitter_microseconds')
cycle('Jitter as relative amplitude perturbation (r.a.p.)', 'Jitter_rap')
cycle('Jitter as 5-point period perturbation quotient (p.p.q.5)', 'Jitter_ppq5')
cycle('Jitter as average absolute difference of differences between jitter cycles (d.d.p.)', 'Jitter_ddp')
cycle('Shimmer in %', 'Shimmer_percent')
cycle('Absolute shimmer in decibels (dB)', 'Shimmer_db')
cycle('Shimmer as 3-point amplitude perturbation quotient (a.p.q.3)', 'Shimmer_apq3')
cycle('Shimmer as 5-point amplitude perturbation quotient (a.p.q.5)', 'Shimmer_apq5')
cycle('Shimmer as 11-point amplitude perturbation quotient (a.p.q.11)', 'Shimmer_apq11')
cycle('Shimmer as average absolute differences between consecutive differences between the amplitudes of shimmer cycles (d.d.a.)', 'Shimmer_dda')
cycle('Autocorrelation between NHR and HNR', 'Harmonicity_NHR_vs_HNR')
cycle('Noise-to-Harmonic ratio (NHR)', 'Harmonicity_NHR')
cycle('Harmonic-to-Noise ratio (HNR)', 'Harmonicity_HNR')
cycle('Median pitch', 'Pitch_median')
cycle('Mean pitch', 'Pitch_mean')
cycle('Standard deviation of pitch', 'Pitch_SD')
cycle('Minimum pitch', 'Pitch_min')
cycle('Maximum pitch', 'Pitch_max')
cycle('Number of pulses', 'Pulse_no_pulses')
cycle('Number of periods', 'Pulse_no_periods')
cycle('Mean period', 'Pulse_mean')
cycle('Standard deviation of period', 'Pulse_SD')
cycle('Fraction of unvoiced frames', 'Voice_fuf')
cycle('Degree of voice breaks', 'Voice_no_breaks')
cycle('Absolute jitter in microseconds', 'Voice_deg_Breaks')