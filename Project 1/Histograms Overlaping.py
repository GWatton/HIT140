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
def cycle (a,b,c,d,e):
    #Print summary statistics of the "Jitter in %" column only
    print(c, "with Parkinsons:", withpark[d].describe())
    print(c, "without Parkinsons:", withoutpark[d].describe())

    #Extract the column, e.g. "Jitter in %", values as a numpy array (this excludes the column name)
    sample = a
    sample2 = b

    #Get bin width of sample
    max_val = sample.max()
    min_val = sample.min()
    the_range = max_val - min_val
    bin_width = e
    bin_count= int(the_range/bin_width)
    
    #Plot histograms of data with overlap from people with and without Parkinsons
    plt.hist(sample, alpha=0.5, color='blue', edgecolor='black', bins=bin_count, label='With Parkinsons')
    plt.hist(sample2, alpha=0.5, color='red', edgecolor='black', bins=bin_count, label='Without Parkinsons') 
    plt.legend(loc='upper right')
    plt.title('Parkinsons VS Non-Parkinsons Voice Samples')
    plt.xlabel(c)
    plt.ylabel('QTY')
    plt.show()

#This runs the function for each column within the data set and compares the results of each column in an overlay
cycle(withpark.Jitter_percent.values, withoutpark.Jitter_percent.values, 'Jitter in %', 'Jitter_percent', 0.5)
cycle(withpark.Jitter_microseconds.values, withoutpark.Jitter_microseconds.values, 'Absolute jitter in microseconds', 'Jitter_microseconds', 0.00005)
cycle(withpark.Jitter_rap.values, withoutpark.Jitter_rap.values, 'Jitter as relative amplitude perturbation (r.a.p.)', 'Jitter_rap', 0.1)
cycle(withpark.Jitter_ppq5.values, withoutpark.Jitter_ppq5.values, 'Jitter as 5-point period perturbation quotient (p.p.q.5)', 'Jitter_ppq5', 0.1)
cycle(withpark.Jitter_ddp.values, withoutpark.Jitter_ddp.values, 'Jitter as average absolute difference of differences between jitter cycles (d.d.p.)', 'Jitter_ddp', 0.5)
cycle(withpark.Shimmer_percent.values, withoutpark.Shimmer_percent.values, 'Shimmer in %', 'Shimmer_percent', 1)
cycle(withpark.Shimmer_db.values, withoutpark.Shimmer_db.values, 'Absolute shimmer in decibels (dB)', 'Shimmer_db', 0.1)
cycle(withpark.Shimmer_apq3.values, withoutpark.Shimmer_apq3.values, 'Shimmer as 3-point amplitude perturbation quotient (a.p.q.3)', 'Shimmer_apq3', 0.5)
cycle(withpark.Shimmer_apq5.values, withoutpark.Shimmer_apq5.values, 'Shimmer as 5-point amplitude perturbation quotient (a.p.q.5)', 'Shimmer_apq5', 0.5)
cycle(withpark.Shimmer_apq11.values, withoutpark.Shimmer_apq11.values, 'Shimmer as 11-point amplitude perturbation quotient (a.p.q.11)', 'Shimmer_apq11', 1)
cycle(withpark.Shimmer_dda.values, withoutpark.Shimmer_dda.values, 'Shimmer as average absolute differences between consecutive differences between the amplitudes of shimmer cycles (d.d.a.)', 'Shimmer_dda', 1)
cycle(withpark.Harmonicity_NHR_vs_HNR.values, withoutpark.Harmonicity_NHR_vs_HNR.values, 'Autocorrelation between NHR and HNR', 'Harmonicity_NHR_vs_HNR', 0.05)
cycle(withpark.Harmonicity_NHR.values, withoutpark.Harmonicity_NHR.values, 'Noise-to-Harmonic ratio (NHR)', 'Harmonicity_NHR', 0.05)
cycle(withpark.Harmonicity_HNR.values, withoutpark.Harmonicity_HNR.values, 'Harmonic-to-Noise ratio (HNR)', 'Harmonicity_HNR', 1)
cycle(withpark.Pitch_median.values, withoutpark.Pitch_median.values, 'Median pitch', 'Pitch_median', 50)
cycle(withpark.Pitch_mean.values, withoutpark.Pitch_mean.values, 'Mean pitch', 'Pitch_mean', 50)
cycle(withpark.Pitch_SD.values, withoutpark.Pitch_SD.values, 'Standard deviation of pitch', 'Pitch_SD', 10)
cycle(withpark.Pitch_min.values, withoutpark.Pitch_min.values, 'Minimum pitch', 'Pitch_min', 50)
cycle(withpark.Pitch_max.values, withoutpark.Pitch_max.values, 'Maximum pitch', 'Pitch_max', 50)
cycle(withpark.Pulse_no_pulses.values, withoutpark.Pulse_no_pulses.values, 'Number of pulses', 'Pulse_no_pulses', 50)
cycle(withpark.Pulse_no_periods.values, withoutpark.Pulse_no_periods.values, 'Number of periods', 'Pulse_no_periods', 50)
cycle(withpark.Pulse_mean.values, withoutpark.Pulse_mean.values, 'Mean period', 'Pulse_mean', 0.0005)
cycle(withpark.Pulse_SD.values, withoutpark.Pulse_SD.values, 'Standard deviation of period', 'Pulse_SD', 0.00005)
cycle(withpark.Voice_fuf.values, withoutpark.Voice_fuf.values, 'Fraction of unvoiced frames', 'Voice_fuf', 5)
cycle(withpark.Voice_no_breaks.values, withoutpark.Voice_no_breaks.values, 'Degree of voice breaks', 'Voice_no_breaks', 1)
cycle(withpark.Voice_deg_Breaks.values, withoutpark.Voice_deg_Breaks.values, 'Absolute jitter in microseconds', 'Voice_deg_Breaks', 5)