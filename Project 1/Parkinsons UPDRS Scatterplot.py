import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st

# Column names to be added
column_names=['Subject_identifier','Jitter_percent','Jitter_microseconds','Jitter_rap','Jitter_ppq5','Jitter_ddp','Shimmer_percent','Shimmer_db','Shimmer_apq3','Shimmer_apq5','Shimmer_apq11','Shimmer_dda','Harmonicity_NHR_vs_HNR','Harmonicity_NHR','Harmonicity_HNR','Pitch_median','Pitch_mean','Pitch_SD','Pitch_min','Pitch_max','Pulse_no_pulses','Pulse_no_periods','Pulse_mean','Pulse_SD','Voice_fuf','Voice_no_breaks','Voice_deg_Breaks','UPDRS','PD_indicator']

#Reads data files and adds columns to data file
df = pd.read_csv('po1_data.txt', names=column_names)
withoutpark = df[df["PD_indicator"]==0]


xplot = []
yplot = []


def scat(a,b):
    vertline = st.tmean(withoutpark[a])

    N = 20
    x3 = xplot
    y3 = yplot
    colors = np.random.rand(N)
    area = (30 * np.random.rand(N))**2 
    plt.title(b)
    plt.ylabel("UPDRS Rating")
    plt.scatter(x3, y3, s=50, c=colors)
    plt.axvline(x = vertline, color = 'b', label = 'axvline - full height')
    plt.show()
    xplot.clear()
    yplot.clear()


for x in range(1,21):
    working = df[df["Subject_identifier"]==x]
    updrs = working.UPDRS.values
    updrs1 = st.tmean(updrs)
    yplot.append(int(updrs1))
    def cycle(a):
        working1 = a
        x_bar = st.tmean(working1)
        xplot.append(float(x_bar))
    cycle(working.Jitter_percent.values)

scat('Jitter_percent','Jitter in %')


for x in range(1,21):
    working = df[df["Subject_identifier"]==x]
    updrs = working.UPDRS.values
    updrs1 = st.tmean(updrs)
    yplot.append(int(updrs1))
    def cycle(a):
        working1 = a
        x_bar = st.tmean(working1)
        xplot.append(float(x_bar))
    cycle(working.Jitter_microseconds.values)

scat('Jitter_microseconds', 'Absolute jitter in microseconds')


for x in range(1,21):
    working = df[df["Subject_identifier"]==x]
    updrs = working.UPDRS.values
    updrs1 = st.tmean(updrs)
    yplot.append(int(updrs1))
    def cycle(a):
        working1 = a
        x_bar = st.tmean(working1)
        xplot.append(float(x_bar))
    cycle(working.Jitter_rap.values)

scat('Jitter_rap', 'Jitter as relative amplitude perturbation (r.a.p.)')


for x in range(1,21):
    working = df[df["Subject_identifier"]==x]
    updrs = working.UPDRS.values
    updrs1 = st.tmean(updrs)
    yplot.append(int(updrs1))
    def cycle(a):
        working1 = a
        x_bar = st.tmean(working1)
        xplot.append(float(x_bar))
    cycle(working.Jitter_ppq5.values)

scat('Jitter_ppq5', 'Jitter as 5-point period perturbation quotient (p.p.q.5)')


for x in range(1,21):
    working = df[df["Subject_identifier"]==x]
    updrs = working.UPDRS.values
    updrs1 = st.tmean(updrs)
    yplot.append(int(updrs1))
    def cycle(a):
        working1 = a
        x_bar = st.tmean(working1)
        xplot.append(float(x_bar))
    cycle(working.Jitter_ddp.values)

scat('Jitter_ddp', 'Jitter as average absolute difference of differences between jitter cycles (d.d.p.)')


for x in range(1,21):
    working = df[df["Subject_identifier"]==x]
    updrs = working.UPDRS.values
    updrs1 = st.tmean(updrs)
    yplot.append(int(updrs1))
    def cycle(a):
        working1 = a
        x_bar = st.tmean(working1)
        xplot.append(float(x_bar))
    cycle(working.Shimmer_percent.values)

scat('Shimmer_percent', 'Shimmer in %')


for x in range(1,21):
    working = df[df["Subject_identifier"]==x]
    updrs = working.UPDRS.values
    updrs1 = st.tmean(updrs)
    yplot.append(int(updrs1))
    def cycle(a):
        working1 = a
        x_bar = st.tmean(working1)
        xplot.append(float(x_bar))
    cycle(working.Shimmer_apq3.values)

scat('Shimmer_db','Absolute shimmer in decibels (dB)')

for x in range(1,21):
    working = df[df["Subject_identifier"]==x]
    updrs = working.UPDRS.values
    updrs1 = st.tmean(updrs)
    yplot.append(int(updrs1))
    def cycle(a):
        working1 = a
        x_bar = st.tmean(working1)
        xplot.append(float(x_bar))
    cycle(working.Shimmer_apq3.values)

scat('Shimmer_apq3', 'Shimmer as 3-point amplitude perturbation quotient (a.p.q.3)')


for x in range(1,21):
    working = df[df["Subject_identifier"]==x]
    updrs = working.UPDRS.values
    updrs1 = st.tmean(updrs)
    yplot.append(int(updrs1))
    def cycle(a):
        working1 = a
        x_bar = st.tmean(working1)
        xplot.append(float(x_bar))
    cycle(working.Shimmer_apq5.values)

scat('Shimmer_apq5', 'Shimmer as 5-point amplitude perturbation quotient (a.p.q.5)')


for x in range(1,21):
    working = df[df["Subject_identifier"]==x]
    updrs = working.UPDRS.values
    updrs1 = st.tmean(updrs)
    yplot.append(int(updrs1))
    def cycle(a):
        working1 = a
        x_bar = st.tmean(working1)
        xplot.append(float(x_bar))
    cycle(working.Shimmer_apq11.values)

scat('Shimmer_apq11', 'Shimmer as 11-point amplitude perturbation quotient (a.p.q.11)' )


for x in range(1,21):
    working = df[df["Subject_identifier"]==x]
    updrs = working.UPDRS.values
    updrs1 = st.tmean(updrs)
    yplot.append(int(updrs1))
    def cycle(a):
        working1 = a
        x_bar = st.tmean(working1)
        xplot.append(float(x_bar))
    cycle(working.Shimmer_dda.values)

scat('Shimmer_dda', 'Shimmer as average absolute differences between consecutive differences between the amplitudes of shimmer cycles (d.d.a.)')


for x in range(1,21):
    working = df[df["Subject_identifier"]==x]
    updrs = working.UPDRS.values
    updrs1 = st.tmean(updrs)
    yplot.append(int(updrs1))
    def cycle(a):
        working1 = a
        x_bar = st.tmean(working1)
        xplot.append(float(x_bar))
    cycle(working.Harmonicity_NHR_vs_HNR.values)

scat('Harmonicity_NHR_vs_HNR', 'Autocorrelation between NHR and HNR')


for x in range(1,21):
    working = df[df["Subject_identifier"]==x]
    updrs = working.UPDRS.values
    updrs1 = st.tmean(updrs)
    yplot.append(int(updrs1))
    def cycle(a):
        working1 = a
        x_bar = st.tmean(working1)
        xplot.append(float(x_bar))
    cycle(working.Harmonicity_NHR.values)

scat('Harmonicity_NHR', 'Noise-to-Harmonic ratio (NHR)')


for x in range(1,21):
    working = df[df["Subject_identifier"]==x]
    updrs = working.UPDRS.values
    updrs1 = st.tmean(updrs)
    yplot.append(int(updrs1))
    def cycle(a):
        working1 = a
        x_bar = st.tmean(working1)
        xplot.append(float(x_bar))
    cycle(working.Harmonicity_HNR.values)

scat('Harmonicity_HNR', 'Harmonic-to-Noise ratio (HNR)')


for x in range(1,21):
    working = df[df["Subject_identifier"]==x]
    updrs = working.UPDRS.values
    updrs1 = st.tmean(updrs)
    yplot.append(int(updrs1))
    def cycle(a):
        working1 = a
        x_bar = st.tmean(working1)
        xplot.append(float(x_bar))
    cycle(working.Pitch_median.values)

scat('Pitch_median', 'Median pitch')


for x in range(1,21):
    working = df[df["Subject_identifier"]==x]
    updrs = working.UPDRS.values
    updrs1 = st.tmean(updrs)
    yplot.append(int(updrs1))
    def cycle(a):
        working1 = a
        x_bar = st.tmean(working1)
        xplot.append(float(x_bar))
    cycle(working.Pitch_mean.values)

scat('Pitch_mean', 'Mean pitch')


for x in range(1,21):
    working = df[df["Subject_identifier"]==x]
    updrs = working.UPDRS.values
    updrs1 = st.tmean(updrs)
    yplot.append(int(updrs1))
    def cycle(a):
        working1 = a
        x_bar = st.tmean(working1)
        xplot.append(float(x_bar))
    cycle(working.Pitch_SD.values)

scat('Pitch_SD', 'Standard deviation of pitch')


for x in range(1,21):
    working = df[df["Subject_identifier"]==x]
    updrs = working.UPDRS.values
    updrs1 = st.tmean(updrs)
    yplot.append(int(updrs1))
    def cycle(a):
        working1 = a
        x_bar = st.tmean(working1)
        xplot.append(float(x_bar))
    cycle(working.Pitch_min.values)

scat('Pitch_min', 'Minimum pitch')


for x in range(1,21):
    working = df[df["Subject_identifier"]==x]
    updrs = working.UPDRS.values
    updrs1 = st.tmean(updrs)
    yplot.append(int(updrs1))
    def cycle(a):
        working1 = a
        x_bar = st.tmean(working1)
        xplot.append(float(x_bar))
    cycle(working.Pitch_max.values)

scat('Pitch_max', 'Maximum pitch')


for x in range(1,21):
    working = df[df["Subject_identifier"]==x]
    updrs = working.UPDRS.values
    updrs1 = st.tmean(updrs)
    yplot.append(int(updrs1))
    def cycle(a):
        working1 = a
        x_bar = st.tmean(working1)
        xplot.append(float(x_bar))
    cycle(working.Pulse_no_pulses.values)

scat('Pulse_no_pulses', 'Number of pulses')


for x in range(1,21):
    working = df[df["Subject_identifier"]==x]
    updrs = working.UPDRS.values
    updrs1 = st.tmean(updrs)
    yplot.append(int(updrs1))
    def cycle(a):
        working1 = a
        x_bar = st.tmean(working1)
        xplot.append(float(x_bar))
    cycle(working.Pulse_no_periods.values)

scat('Pulse_no_periods', 'Number of periods')


for x in range(1,21):
    working = df[df["Subject_identifier"]==x]
    updrs = working.UPDRS.values
    updrs1 = st.tmean(updrs)
    yplot.append(int(updrs1))
    def cycle(a):
        working1 = a
        x_bar = st.tmean(working1)
        xplot.append(float(x_bar))
    cycle(working.Pulse_mean.values)

scat('Pulse_mean', 'Mean period')


for x in range(1,21):
    working = df[df["Subject_identifier"]==x]
    updrs = working.UPDRS.values
    updrs1 = st.tmean(updrs)
    yplot.append(int(updrs1))
    def cycle(a):
        working1 = a
        x_bar = st.tmean(working1)
        xplot.append(float(x_bar))
    cycle(working.Pulse_SD.values)

scat('Pulse_SD', 'Standard deviation of period')


for x in range(1,21):
    working = df[df["Subject_identifier"]==x]
    updrs = working.UPDRS.values
    updrs1 = st.tmean(updrs)
    yplot.append(int(updrs1))
    def cycle(a):
        working1 = a
        x_bar = st.tmean(working1)
        xplot.append(float(x_bar))
    cycle(working.Voice_fuf.values)

scat('Voice_fuf', 'Fraction of unvoiced frames')


for x in range(1,21):
    working = df[df["Subject_identifier"]==x]
    updrs = working.UPDRS.values
    updrs1 = st.tmean(updrs)
    yplot.append(int(updrs1))
    def cycle(a):
        working1 = a
        x_bar = st.tmean(working1)
        xplot.append(float(x_bar))
    cycle(working.Voice_no_breaks.values)

scat('Voice_no_breaks', 'Number of voice breaks')


for x in range(1,21):
    working = df[df["Subject_identifier"]==x]
    updrs = working.UPDRS.values
    updrs1 = st.tmean(updrs)
    yplot.append(int(updrs1))
    def cycle(a):
        working1 = a
        x_bar = st.tmean(working1)
        xplot.append(float(x_bar))
    cycle(working.Voice_deg_Breaks.values)

scat('Voice_deg_Breaks', 'Degree of voice breaks')