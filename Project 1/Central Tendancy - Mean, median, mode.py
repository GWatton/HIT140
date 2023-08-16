import scipy.stats as st
import statistics as sta
import numpy as np
import pandas as pd


# Column names to be added
column_names=['Subject_identifier','Jitter_percent','Jitter_microseconds','Jitter_rap','Jitter_ppq5','Jitter_ddp','Shimmer_percent','Shimmer_db','Shimmer_apq3','Shimmer_apq5','Shimmer_apq11','Shimmer_dda','Harmonicity_NHR_vs_HNR','Harmonicity_NHR','Harmonicity_HNR','Pitch_median','Pitch_mean','Pitch_SD','Pitch_min','Pitch_max','Pulse_no_pulses','Pulse_no_periods','Pulse_mean','Pulse_SD','Voice_fuf','Voice_no_breaks','Voice_deg_Breaks','UPDRS','PD_indicator']

#Reads data files and adds columns to data file
df = pd.read_csv('po1_data.txt', names=column_names)

#Seperates data files into seperate files if the subject has Parkinsons (PD Indicator of 1) or has no Parkinsons (PD Indicator of 0)
withparkinsons = df[df["PD_indicator"]==1]
withoutpark = df[df["PD_indicator"]==0]

#Function to calculate the mean, mode and median of people with Parkinsons vs without Parkinsons
def cycle(a, b, c):
    withpark = a
    nopark= b

    #Calculates the mean, mode, and median of people with Parkinsons
    wparkmean = st.tmean(withpark)
    wparkmode = sta.mode(withpark)
    wparkmedian = sta.median(withpark) 

    #Calculates the mean, mode, and median of people without Parkinsons
    woparkmean = st.tmean(nopark)
    woparkmode = sta.mode(nopark)
    woparkmedian = sta.median(nopark)
    
    #Formats and prints the outputs
    print('    ','Without Parkinsons data:','''
    ''',c, 'mean:', woparkmean,'''
    ''',c, 'mode:', woparkmode, '''
    ''',c, 'median:', woparkmedian,'''
    ''','With Parkinsons:','''
    ''',c, 'mean:', wparkmean, '''
    ''',c, 'mode:', wparkmode, '''
    ''',c, 'median:', wparkmedian,'''
    ''' )

#Runs the program for each column
cycle(withoutpark.Jitter_percent.values, withparkinsons.Jitter_percent.values, 'Jitter in %')
cycle(withoutpark.Jitter_microseconds.values, withparkinsons.Jitter_microseconds.values, 'Absolute jitter in microseconds' )
cycle(withoutpark.Jitter_rap.values, withparkinsons.Jitter_rap.values, 'Jitter as relative amplitude perturbation (r.a.p.)' )
cycle(withoutpark.Jitter_ppq5.values, withparkinsons.Jitter_ppq5.values, 'Jitter as 5-point period perturbation quotient (p.p.q.5)' )
cycle(withoutpark.Jitter_ddp.values, withparkinsons.Jitter_ddp.values, 'Jitter as average absolute difference of differences between jitter cycles (d.d.p.)' )
cycle(withoutpark.Shimmer_percent.values, withparkinsons.Shimmer_percent.values, 'Shimmer in %' )
cycle(withoutpark.Shimmer_db.values, withparkinsons.Shimmer_db.values, 'Absolute shimmer in decibels (dB)' )
cycle(withoutpark.Shimmer_apq3.values, withparkinsons.Shimmer_apq3.values, 'Shimmer as 3-point amplitude perturbation quotient (a.p.q.3)' )
cycle(withoutpark.Shimmer_apq5.values, withparkinsons.Shimmer_apq5.values, 'Shimmer as 5-point amplitude perturbation quotient (a.p.q.5)' )
cycle(withoutpark.Shimmer_apq11.values, withparkinsons.Shimmer_apq11.values, 'Shimmer as 11-point amplitude perturbation quotient (a.p.q.11)')
cycle(withoutpark.Shimmer_dda.values, withparkinsons.Shimmer_dda.values, 'Shimmer as average absolute differences between consecutive differences between the amplitudes of shimmer cycles (d.d.a.)' )
cycle(withoutpark.Harmonicity_NHR_vs_HNR.values, withparkinsons.Harmonicity_NHR_vs_HNR.values, 'Autocorrelation between NHR and HNR' )
cycle(withoutpark.Harmonicity_NHR.values, withparkinsons.Harmonicity_NHR.values, 'Noise-to-Harmonic ratio (NHR)' )
cycle(withoutpark.Harmonicity_HNR.values, withparkinsons.Harmonicity_HNR.values, 'Harmonic-to-Noise ratio (HNR)' )
cycle(withoutpark.Pitch_median.values, withparkinsons.Pitch_median.values, 'Median pitch' )
cycle(withoutpark.Pitch_mean.values, withparkinsons.Pitch_mean.values, 'Mean pitch' )
cycle(withoutpark.Pitch_SD.values, withparkinsons.Pitch_SD.values, 'Standard deviation of pitch' )
cycle(withoutpark.Pitch_min.values, withparkinsons.Pitch_min.values, 'Minimum pitch' )
cycle(withoutpark.Pitch_max.values, withparkinsons.Pitch_max.values, 'Maximum pitch' )
cycle(withoutpark.Pulse_no_pulses.values, withparkinsons.Pulse_no_pulses.values, 'Number of pulses' )
cycle(withoutpark.Pulse_no_periods.values, withparkinsons.Pulse_no_periods.values, 'Number of periods' )
cycle(withoutpark.Pulse_mean.values, withparkinsons.Pulse_mean.values, 'Mean period' )
cycle(withoutpark.Pulse_SD.values, withparkinsons.Pulse_SD.values, 'Standard deviation of period' )
cycle(withoutpark.Voice_fuf.values, withparkinsons.Voice_fuf.values, 'Fraction of unvoiced frames' )
cycle(withoutpark.Voice_no_breaks.values, withparkinsons.Voice_no_breaks.values, 'Degree of voice breaks' )
cycle(withoutpark.Voice_deg_Breaks.values, withparkinsons.Voice_deg_Breaks.values, 'Absolute jitter in microseconds' )    

