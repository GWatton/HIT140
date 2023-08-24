import statsmodels.stats.weightstats as stm
import scipy.stats as st
import statistics as sta
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns

print('\n\n')
print('#    #   #   #########   #       #   #####         GGGGGGG   RRRRRRR  OOOOOOO  U      U  PPPPPPP    6666666 ')
print('#    #           #       #      ##   #   #         G         R     R  O     O  U      U  P     P    6')
print('#    #   #       #       #     # #   #   #         G         R     R  O     O  U      U  P     P    6')
print('######   #       #       #    #  #   #   #         G         RRRRRRR  O     O  U      U  PPPPPPP    6666666')
print('#    #   #       #       #   ######  #   #         G    GG   R   R    O     O  U      U  P          6     6')
print('#    #   #       #       #       #   #   #         G     G   R    R   O     O  U      U  P          6     6')
print('#    #   #       #       #       #   #####         GGGGGGG   R     R  OOOOOOO  UUUUUUUU  P          6666666')
print('\n\n')
print('\n##########  Group 6 Parkinsons data analysis ##########\n')
userchoice = 1
while userchoice != 0:
    print('\nPlease select the data you would like to view: \n\n', 'Select 1 for: Correlation Matrix & Scatterplots  \n Select 2 for: Confidence Interval \n Select 3 for: Histograms \n Select 4 for: Two Sample T Test\n Select 5 for: Central Tendancy - Mean, Median, Mode \n Select 0 to Quit\n')
    
    
    userchoice = input("\n Selection: ")
    print('\n')


    if int(userchoice) == 1:
        
        # Open the file
        with open('po1_data.txt', 'r') as file:
            # Read the contents of the file
            data = file.readlines()

        # Remove any leading or trailing whitespace from each line
        data = [line.strip() for line in data]

        # Specify the column names
        column_names = ['Subject_no.', 'Jitter(%)', 'jitter(abs)', 'jitter(rap)', 'jitter(ppq5)', 'jitter(ddp)', 'shimmer(%)', 'shimmer(abs)', 'shimmer(apq3)', 'shimmer(apq5)', 'shimmer(apq11)', 'shimmer(dda)', 'anhr&hnr', 'nhr', 'hnr', 'meadian_pitch', 'mean_pitch', 'SD_pitch', 'min_pitch', 'max_pitch', 'no.pulse', 'no.period', 'mean_period', 'sd_period', 'frac_UVF', 'NO.VB', 'Deg.VB', 'updrs', 'status']

        # Create the DataFrame
        df = pd.DataFrame([line.split(',') for line in data], columns=column_names)
        df.describe()
        df.info()

        # Convert columns to float64
        float_columns = ['Jitter(%)', 'jitter(abs)', 'jitter(rap)', 'jitter(ppq5)', 'jitter(ddp)', 'shimmer(%)', 'shimmer(abs)', 'shimmer(apq3)', 'shimmer(apq5)', 'shimmer(apq11)', 'shimmer(dda)', 'anhr&hnr', 'nhr', 'hnr', 'meadian_pitch', 'mean_pitch', 'SD_pitch', 'min_pitch', 'max_pitch', 'no.pulse', 'no.period', 'mean_period', 'sd_period', 'frac_UVF', 'NO.VB', 'Deg.VB', 'updrs']
        df[float_columns] = df[float_columns].astype(float)

        # Convert columns to int64
        int_columns = ['Subject_no.', 'status']
        df[int_columns] = df[int_columns].astype(int)


        df.info()


        df.describe()


        df.shape


        df.status.value_counts()


        df1 = df.groupby('Subject_no.').mean().reset_index()

        #drop the subject no. column from the dataset
        df1.drop('Subject_no.', axis=1, inplace=True)

        # Compute the correlation matrix
        correlation_matrix = df1.corr()

        # Get the absolute correlation values with respect to the "status" column
        correlation_with_status = correlation_matrix['status'].abs().sort_values(ascending=False)

        # Print the salient features
        salient_features = correlation_with_status[correlation_with_status > 0.3].index
        print(salient_features)


        # Plot the correlation heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Heatmap')
        plt.show()


        # Calculate Spearman's rank correlation coefficient
        from scipy.stats import spearmanr
        correlation_coefficients = {}
        for column in df1.columns:
            if column != 'status':
                coefficient, _ = spearmanr(df1[column], df1['status'])
                correlation_coefficients[column] = coefficient

        # Sort the correlation coefficients in descending order
        sorted_coefficients = sorted(correlation_coefficients.items(), key=lambda x: abs(x[1]), reverse=True)

        # Print the correlation coefficients
        for column, coefficient in sorted_coefficients:
            print(f"{column}: {coefficient}")


        #Create a scatter plot for updrs
        plt.scatter(df1['status'], df1['updrs'])
        plt.xlabel('status')
        plt.ylabel('updrs')
        plt.title('Scatter Plot')
        plt.show()


        #Create a scatter plot Degree of voice breaks
        plt.scatter(df1['status'], df1['Deg.VB'])
        plt.xlabel('status')
        plt.ylabel('Deg.VB')
        plt.title('Scatter Plot')
        plt.show()


        #Create a scatter plot No of voice breaks
        plt.scatter(df1['status'], df1['NO.VB'])
        plt.xlabel('status')
        plt.ylabel('NO.VB')
        plt.title('Scatter Plot')
        plt.show()


        #Attributes in the Group
        Atr1='updrs'
        Atr2='Deg.VB'
        Atr3='NO.VB'


        ##EDA: Correlation of attributes of group with other attributes.
        corr_atr1=df1[df1.columns].corr()[Atr1][:]
        corr_atr2=df1[df1.columns].corr()[Atr2][:]
        corr_atr3=df1[df1.columns].corr()[Atr3][:]


        pd.concat([round(corr_atr1,4),round(corr_atr2,4),round(corr_atr3,4)],axis=1,sort=False).T

    
##################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################    
    elif int(userchoice) == 2:
        # Column names to be added
        column_names=['Subject_identifier','Jitter_percent','Jitter_microseconds','Jitter_rap','Jitter_ppq5','Jitter_ddp','Shimmer_percent','Shimmer_db','Shimmer_apq3','Shimmer_apq5','Shimmer_apq11','Shimmer_dda','Harmonicity_NHR_vs_HNR','Harmonicity_NHR','Harmonicity_HNR','Pitch_median','Pitch_mean','Pitch_SD','Pitch_min','Pitch_max','Pulse_no_pulses','Pulse_no_periods','Pulse_mean','Pulse_SD','Voice_fuf','Voice_no_breaks','Voice_deg_Breaks','UPDRS','PD_indicator']

        #Reads data files and adds columns to data file
        dfcon = pd.read_csv('po1_data.txt', names=column_names)

        #Seperates data files into seperate files if the subject has Parkinsons (PD Indicator of 1) or has no Parkinsons (PD Indicator of 0)
        withparkinsonscon = dfcon[dfcon["PD_indicator"]==1]
        withoutparkcon = dfcon[dfcon["PD_indicator"]==0]

        #Line break to assist with review of information
        print('--------------------------------------------------------------------')

        #Function to find the, mean, standard deviation, size, standard error, degrees of freedom, confidence level, significance level and confidence interval
        def confidence (a,b,c,d):
            sample = a[b].to_numpy()


            print(c,':')
            x_bar = st.tmean(sample)
            s = st.tstd(sample)
            n = len(sample)
            print("Mean: %.2f. Standard deviation: %.2f. Size: %d." % (x_bar, s, n))


            std_err = s / math.sqrt(n)
            print("Standard error: %.2f" % std_err)


            df = n - 1
            print("Degrees of freedom: %d" % df)


            conf_lvl = 0.95
            print("Confidence level: %.2f" % conf_lvl)


            sig_lvl = 1 - conf_lvl
            print("Significance level: %.2f" % sig_lvl)


            ci_low_stm, ci_upp_stm = stm._tconfint_generic(x_bar,std_err,df, alpha=sig_lvl, alternative="two-sided")
            print("C.I. of the mean at %d%% confidence level is between %.2f and %.2f." % 
                    (conf_lvl*100, ci_low_stm, ci_upp_stm), '\n',d)

        #Second function the same as the above function, but for columns with decimal places out to 9
        def decimal (a,b,c,d):
            sample = a[b].to_numpy()


            print(c,':')
            x_bar = st.tmean(sample)
            s = st.tstd(sample)
            n = len(sample)
            print("Mean: %.9f. Standard deviation: %.9f. Size: %d." % (x_bar, s, n))


            std_err = s / math.sqrt(n)
            print("Standard error: %.9f" % std_err)


            df = n - 1
            print("Degrees of freedom: %d" % df)


            conf_lvl = 0.95
            print("Confidence level: %.2f" % conf_lvl)


            sig_lvl = 1 - conf_lvl
            print("Significance level: %.2f" % sig_lvl)


            ci_low_stm, ci_upp_stm = stm._tconfint_generic(x_bar,std_err,df, alpha=sig_lvl, alternative="two-sided")
            print("C.I. of the mean at %d%% confidence level is between %.9f and %.9f." % 
                    (conf_lvl*100, ci_low_stm, ci_upp_stm), '\n', d)


        confidencechoice = 1
        while confidencechoice != 0:
            print ("To view a columns confidence interval\n", 'Select 1 for: Jitter in % \n',
                'Select 2 for: Absolute jitter in microseconds \n',
                'Select 3 for: Jitter as relative amplitude perturbation (r.a.p.) \n',
                'Select 4 for: Jitter as 5-point period perturbation quotient (p.p.q.5) \n',
                'Select 5 for: Jitter as average absolute difference of differences between jitter confidences (d.d.p.) \n',
                'Select 6 for: Shimmer in % \n',
                'Select 7 for: Absolute shimmer in decibels (dB) \n',
                'Select 8 for: Shimmer as 3-point amplitude perturbation quotient (a.p.q.3) \n',
                'Select 9 for: Shimmer as 5-point amplitude perturbation quotient (a.p.q.5)\n',
                'Select 10 for: Shimmer as 11-point amplitude perturbation quotient (a.p.q.11) \n',
                'Select 11 for: Shimmer as average absolute differences between consecutive differences between the amplitudes of shimmer confidences (d.d.a.) \n',
                'Select 12 for: Autocorrelation between NHR and HNR \n',
                'Select 13 for: Noise-to-Harmonic ratio (NHR) \n',
                'Select 14 for: Harmonic-to-Noise ratio (HNR) \n',
                'Select 15 for: Median pitch \n',
                'Select 16 for: Mean pitch \n',
                'Select 17 for: Standard deviation of pitch \n',
                'Select 18 for: Minimum Pitch \n',
                'Select 19 for: Maximum Pitch \n',
                'Select 20 for: Number of Pulses \n',
                'Select 21 for: Numer of periods \n',
                'Select 22 for: Mean Pulse \n',
                'Select 23 for: Standard deviation of period \n',
                'Select 24 for: Fraction of unvoiced frames \n',
                'Select 25 for: Number of voice breaks \n',
                'Select 26 for: Degree of voice breaks \n \n',
                'Or select 0 to return to Main Menu\n')
            
            confidencechoice = input("Selection: ")
            print('\n -------------------------------------------------------------------- \n')

            if int(confidencechoice) == 1:

                confidence(withoutparkcon,'Jitter_percent', 'Jitter in %','')
                confidence(withparkinsonscon, 'Jitter_percent', 'Jitter in %','--------------------------------------------------------------------\n')

            elif int(confidencechoice) == 2:
                decimal(withoutparkcon,'Jitter_microseconds', 'Absolute jitter in microseconds','')
                decimal(withparkinsonscon, 'Jitter_microseconds', 'Absolute jitter in microseconds','--------------------------------------------------------------------\n')

            elif int(confidencechoice) == 3:
                confidence(withoutparkcon,'Jitter_rap', 'Jitter as relative amplitude perturbation (r.a.p.)','')
                confidence(withparkinsonscon,'Jitter_rap', 'Jitter as relative amplitude perturbation (r.a.p.)','--------------------------------------------------------------------\n')

            elif int(confidencechoice) == 4:      
                confidence(withoutparkcon, 'Jitter_ppq5', 'Jitter as 5-point period perturbation quotient (p.p.q.5)','')
                confidence(withparkinsonscon, 'Jitter_ppq5','Jitter as 5-point period perturbation quotient (p.p.q.5)','--------------------------------------------------------------------\n')

            elif int(confidencechoice) == 5:
                confidence(withoutparkcon, 'Jitter_ddp','Jitter as average absolute difference of differences between jitter confidences (d.d.p.)','')
                confidence(withparkinsonscon, 'Jitter_ddp','Jitter as average absolute difference of differences between jitter confidences (d.d.p.)','--------------------------------------------------------------------\n')

            elif int(confidencechoice) == 6:
                confidence(withoutparkcon, 'Shimmer_percent','Shimmer in %','')
                confidence(withparkinsonscon, 'Shimmer_percent','Shimmer in %','--------------------------------------------------------------------\n')

            elif int(confidencechoice) == 7:
                confidence(withoutparkcon,'Shimmer_db', 'Absolute shimmer in decibels (dB)','')
                confidence(withparkinsonscon,'Shimmer_db', 'Absolute shimmer in decibels (dB)','--------------------------------------------------------------------\n')

            elif int(confidencechoice) == 8:
                confidence(withoutparkcon,'Shimmer_apq3', 'Shimmer as 3-point amplitude perturbation quotient (a.p.q.3)','')
                confidence(withparkinsonscon,'Shimmer_apq3', 'Shimmer as 3-point amplitude perturbation quotient (a.p.q.3)','--------------------------------------------------------------------\n')

            elif int(confidencechoice) == 9:
                confidence(withoutparkcon,'Shimmer_apq5', 'Shimmer as 5-point amplitude perturbation quotient (a.p.q.5)','')
                confidence(withparkinsonscon,'Shimmer_apq5', 'Shimmer as 5-point amplitude perturbation quotient (a.p.q.5)','--------------------------------------------------------------------\n')

            elif int(confidencechoice) == 10:
                confidence(withoutparkcon,'Shimmer_apq11', 'Shimmer as 11-point amplitude perturbation quotient (a.p.q.11)','')
                confidence(withparkinsonscon,'Shimmer_apq11', 'Shimmer as 11-point amplitude perturbation quotient (a.p.q.11)','--------------------------------------------------------------------\n')

            elif int(confidencechoice) == 11:
                confidence(withoutparkcon,'Shimmer_dda', 'Shimmer as average absolute differences between consecutive differences between the amplitudes of shimmer confidences (d.d.a.)','')
                confidence(withparkinsonscon,'Shimmer_dda', 'Shimmer as average absolute differences between consecutive differences between the amplitudes of shimmer confidences (d.d.a.)','--------------------------------------------------------------------\n')

            elif int(confidencechoice) == 12:
                decimal(withoutparkcon,'Harmonicity_NHR_vs_HNR', 'Autocorrelation between NHR and HNR','')
                decimal(withparkinsonscon,'Harmonicity_NHR_vs_HNR', 'Autocorrelation between NHR and HNR','--------------------------------------------------------------------\n')

            elif int(confidencechoice) == 13:
                decimal(withoutparkcon,'Harmonicity_NHR', 'Noise-to-Harmonic ratio (NHR)','')
                decimal(withparkinsonscon,'Harmonicity_NHR', 'Noise-to-Harmonic ratio (NHR)','--------------------------------------------------------------------\n')

            elif int(confidencechoice) == 14:
                confidence(withoutparkcon,'Harmonicity_HNR', 'Harmonic-to-Noise ratio (HNR)','')
                confidence(withparkinsonscon,'Harmonicity_HNR', 'Harmonic-to-Noise ratio (HNR)','--------------------------------------------------------------------\n')

            elif int(confidencechoice) == 15:
                confidence(withoutparkcon,'Pitch_median', 'Median pitch','')
                confidence(withparkinsonscon,'Pitch_median', 'Median pitch','--------------------------------------------------------------------\n')

            elif int(confidencechoice) == 16:
                confidence(withoutparkcon,'Pitch_mean', 'Mean pitch','')
                confidence(withparkinsonscon,'Pitch_mean', 'Mean pitch','--------------------------------------------------------------------\n')

            elif int(confidencechoice) == 17:
                confidence(withoutparkcon,'Pitch_SD', 'Standard deviation of pitch','')
                confidence(withparkinsonscon,'Pitch_SD', 'Standard deviation of pitch','--------------------------------------------------------------------\n')

            elif int(confidencechoice) == 18:
                confidence(withoutparkcon,'Pitch_min', 'Minimum pitch','')
                confidence(withparkinsonscon,'Pitch_min', 'Minimum pitch','--------------------------------------------------------------------\n')

            elif int(confidencechoice) == 19:
                confidence(withoutparkcon,'Pitch_max', 'Maximum pitch','')
                confidence(withparkinsonscon,'Pitch_max', 'Maximum pitch','--------------------------------------------------------------------\n')

            elif int(confidencechoice) == 20:
                confidence(withoutparkcon,'Pulse_no_pulses', 'Number of pulses','')
                confidence(withparkinsonscon,'Pulse_no_pulses', 'Number of pulses','--------------------------------------------------------------------\n')

            elif int(confidencechoice) == 21:
                confidence(withoutparkcon,'Pulse_no_periods', 'Number of periods','')
                confidence(withparkinsonscon,'Pulse_no_periods', 'Number of periods','--------------------------------------------------------------------\n' )

            elif int(confidencechoice) == 22:
                confidence(withoutparkcon, 'Pulse_mean', 'Mean period','')
                confidence(withparkinsonscon,'Pulse_mean', 'Mean period','--------------------------------------------------------------------\n' )

            elif int(confidencechoice) == 23:
                decimal(withoutparkcon,'Pulse_SD', 'Standard deviation of period','')
                decimal(withparkinsonscon,'Pulse_SD', 'Standard deviation of period','--------------------------------------------------------------------\n' )

            elif int(confidencechoice) == 24:
                confidence(withoutparkcon,'Voice_fuf', '','')
                confidence(withparkinsonscon,'Voice_fuf', 'Fraction of unvoiced frames','--------------------------------------------------------------------\n' )

            elif int(confidencechoice) == 25:
                confidence(withoutparkcon, 'Voice_no_breaks','Number of voice breaks','')
                confidence(withparkinsonscon,'Voice_no_breaks', 'Number of voice breaks','--------------------------------------------------------------------\n' )

            elif int(confidencechoice) == 26:
                confidence(withoutparkcon,'Voice_deg_Breaks', 'Degree of voice breaks','')
                confidence(withparkinsonscon,'Voice_deg_Breaks', 'Degree of voice breaks','--------------------------------------------------------------------\n')
                                                                        
        
            else:
                print("Main Menu \n")
                break

##################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
    elif int(userchoice) == 3:

        # Column names to be added
        column_names=['Subject_identifier','Jitter_percent','Jitter_microseconds','Jitter_rap','Jitter_ppq5','Jitter_ddp','Shimmer_percent','Shimmer_db','Shimmer_apq3','Shimmer_apq5','Shimmer_apq11','Shimmer_dda','Harmonicity_NHR_vs_HNR','Harmonicity_NHR','Harmonicity_HNR','Pitch_median','Pitch_mean','Pitch_SD','Pitch_min','Pitch_max','Pulse_no_pulses','Pulse_no_periods','Pulse_mean','Pulse_SD','Voice_fuf','Voice_no_breaks','Voice_deg_Breaks','UPDRS','PD_indicator']

        #Reads data files and adds columns to data file
        dfhisto = pd.read_csv('po1_data.txt', names=column_names)

        #Seperates data files into seperate files if the subject has Parkinsons (PD Indicator of 1) or has no Parkinsons (PD Indicator of 0)
        withparkhisto = dfhisto[dfhisto["PD_indicator"]==1]
        withoutparkhisto = dfhisto[dfhisto["PD_indicator"]==0]

        #Function to produce a  histogram for the desired data set
        def histogramoverlap (a,b,c,d,e):
            #Print summary statistics of the "Jitter in %" column only
            print(c, "with Parkinsons:", withparkhisto[d].describe())
            print(c, "without Parkinsons:", withoutparkhisto[d].describe(), '\n', '\n')

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

        histochoice = 1
        while histochoice != 0:
            print ("To view a columns comparitve overlapping Histograms\n", 'Select 1 for: Jitter in % \n',
                'Select 2 for: Absolute jitter in microseconds \n',
                'Select 3 for: Jitter as relative amplitude perturbation (r.a.p.) \n',
                'Select 4 for: Jitter as 5-point period perturbation quotient (p.p.q.5) \n',
                'Select 5 for: Jitter as average absolute difference of differences between jitter confidences (d.d.p.) \n',
                'Select 6 for: Shimmer in % \n',
                'Select 7 for: Absolute shimmer in decibels (dB) \n',
                'Select 8 for: Shimmer as 3-point amplitude perturbation quotient (a.p.q.3) \n',
                'Select 9 for: Shimmer as 5-point amplitude perturbation quotient (a.p.q.5)\n',
                'Select 10 for: Shimmer as 11-point amplitude perturbation quotient (a.p.q.11) \n',
                'Select 11 for: Shimmer as average absolute differences between consecutive differences between the amplitudes of shimmer confidences (d.d.a.) \n',
                'Select 12 for: Autocorrelation between NHR and HNR \n',
                'Select 13 for: Noise-to-Harmonic ratio (NHR) \n',
                'Select 14 for: Harmonic-to-Noise ratio (HNR) \n',
                'Select 15 for: Median pitch \n',
                'Select 16 for: Mean pitch \n',
                'Select 17 for: Standard deviation of pitch \n',
                'Select 18 for: Minimum Pitch \n',
                'Select 19 for: Maximum Pitch \n',
                'Select 20 for: Number of Pulses \n',
                'Select 21 for: Numer of periods \n',
                'Select 22 for: Mean Pulse \n',
                'Select 23 for: Standard deviation of period \n',
                'Select 24 for: Fraction of unvoiced frames \n',
                'Select 25 for: Number of voice breaks \n',
                'Select 26 for: Degree of voice breaks \n \n',
                'Or select 0 to retun to Main Menu \n')
            
            
            histochoice = input("Selection: ")

            if int(histochoice) == 1:
                histogramoverlap(withparkhisto.Jitter_percent.values, withoutparkhisto.Jitter_percent.values, 'Jitter in %', 'Jitter_percent', 0.5)

            elif int(histochoice) == 2:
                histogramoverlap(withparkhisto.Jitter_microseconds.values, withoutparkhisto.Jitter_microseconds.values, 'Absolute jitter in microseconds', 'Jitter_microseconds', 0.00005)

            elif int(histochoice) == 3:
                histogramoverlap(withparkhisto.Jitter_rap.values, withoutparkhisto.Jitter_rap.values, 'Jitter as relative amplitude perturbation (r.a.p.)', 'Jitter_rap', 0.1)

            elif int(histochoice) == 4:      
                histogramoverlap(withparkhisto.Jitter_ppq5.values, withoutparkhisto.Jitter_ppq5.values, 'Jitter as 5-point period perturbation quotient (p.p.q.5)', 'Jitter_ppq5', 0.1)
            
            elif int(histochoice) == 5:
                histogramoverlap(withparkhisto.Jitter_ddp.values, withoutparkhisto.Jitter_ddp.values, 'Jitter as average absolute difference of differences between jitter histogramoverlaps (d.d.p.)', 'Jitter_ddp', 0.5)

            elif int(histochoice) == 6:
                histogramoverlap(withparkhisto.Shimmer_percent.values, withoutparkhisto.Shimmer_percent.values, 'Shimmer in %', 'Shimmer_percent', 1)

            elif int(histochoice) == 7:
                histogramoverlap(withparkhisto.Shimmer_db.values, withoutparkhisto.Shimmer_db.values, 'Absolute shimmer in decibels (dB)', 'Shimmer_db', 0.1)

            elif int(histochoice) == 8:
                histogramoverlap(withparkhisto.Shimmer_apq3.values, withoutparkhisto.Shimmer_apq3.values, 'Shimmer as 3-point amplitude perturbation quotient (a.p.q.3)', 'Shimmer_apq3', 0.5)

            elif int(histochoice) == 9:
                histogramoverlap(withparkhisto.Shimmer_apq5.values, withoutparkhisto.Shimmer_apq5.values, 'Shimmer as 5-point amplitude perturbation quotient (a.p.q.5)', 'Shimmer_apq5', 0.5)

            elif int(histochoice) == 10:
                histogramoverlap(withparkhisto.Shimmer_apq11.values, withoutparkhisto.Shimmer_apq11.values, 'Shimmer as 11-point amplitude perturbation quotient (a.p.q.11)', 'Shimmer_apq11', 1)

            elif int(histochoice) == 11:
                histogramoverlap(withparkhisto.Shimmer_dda.values, withoutparkhisto.Shimmer_dda.values, 'Shimmer as average absolute differences between consecutive differences between the amplitudes of shimmer histogramoverlaps (d.d.a.)', 'Shimmer_dda', 1)

            elif int(histochoice) == 12:
                histogramoverlap(withparkhisto.Harmonicity_NHR_vs_HNR.values, withoutparkhisto.Harmonicity_NHR_vs_HNR.values, 'Autocorrelation between NHR and HNR', 'Harmonicity_NHR_vs_HNR', 0.05)

            elif int(histochoice) == 13:
                histogramoverlap(withparkhisto.Harmonicity_NHR.values, withoutparkhisto.Harmonicity_NHR.values, 'Noise-to-Harmonic ratio (NHR)', 'Harmonicity_NHR', 0.05)

            elif int(histochoice) == 14:
                histogramoverlap(withparkhisto.Harmonicity_HNR.values, withoutparkhisto.Harmonicity_HNR.values, 'Harmonic-to-Noise ratio (HNR)', 'Harmonicity_HNR', 1)

            elif int(histochoice) == 15:
                histogramoverlap(withparkhisto.Pitch_median.values, withoutparkhisto.Pitch_median.values, 'Median pitch', 'Pitch_median', 50)

            elif int(histochoice) == 16:
                histogramoverlap(withparkhisto.Pitch_mean.values, withoutparkhisto.Pitch_mean.values, 'Mean pitch', 'Pitch_mean', 50)

            elif int(histochoice) == 17:
                histogramoverlap(withparkhisto.Pitch_SD.values, withoutparkhisto.Pitch_SD.values, 'Standard deviation of pitch', 'Pitch_SD', 10)

            elif int(histochoice) == 18:
                histogramoverlap(withparkhisto.Pitch_min.values, withoutparkhisto.Pitch_min.values, 'Minimum pitch', 'Pitch_min', 50)

            elif int(histochoice) == 19:
                histogramoverlap(withparkhisto.Pitch_max.values, withoutparkhisto.Pitch_max.values, 'Maximum pitch', 'Pitch_max', 50)

            elif int(histochoice) == 20:
                histogramoverlap(withparkhisto.Pulse_no_pulses.values, withoutparkhisto.Pulse_no_pulses.values, 'Number of pulses', 'Pulse_no_pulses', 50)

            elif int(histochoice) == 21:
                histogramoverlap(withparkhisto.Pulse_no_periods.values, withoutparkhisto.Pulse_no_periods.values, 'Number of periods', 'Pulse_no_periods', 50)

            elif int(histochoice) == 22:
                histogramoverlap(withparkhisto.Pulse_mean.values, withoutparkhisto.Pulse_mean.values, 'Mean period', 'Pulse_mean', 0.0005)

            elif int(histochoice) == 23:
                histogramoverlap(withparkhisto.Pulse_SD.values, withoutparkhisto.Pulse_SD.values, 'Standard deviation of period', 'Pulse_SD', 0.00005)

            elif int(histochoice) == 24:
                histogramoverlap(withparkhisto.Voice_fuf.values, withoutparkhisto.Voice_fuf.values, 'Fraction of unvoiced frames', 'Voice_fuf', 5)

            elif int(histochoice) == 25:
                histogramoverlap(withparkhisto.Voice_no_breaks.values, withoutparkhisto.Voice_no_breaks.values, 'Number of voice breaks', 'Voice_no_breaks', 1)

            elif int(histochoice) == 26:
                histogramoverlap(withparkhisto.Voice_deg_Breaks.values, withoutparkhisto.Voice_deg_Breaks.values, 'Degree of voice breaks', 'Voice_deg_Breaks', 5)
            
            else:
                print("Main Menu \n")
                break
    
################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################## 
    elif int(userchoice) == 4:      
        # Copy your code for question 4 here
        # Column names to be added
        column_names=['Subject_identifier','Jitter_percent','Jitter_microseconds','Jitter_rap','Jitter_ppq5','Jitter_ddp','Shimmer_percent','Shimmer_db','Shimmer_apq3','Shimmer_apq5','Shimmer_apq11','Shimmer_dda','Harmonicity_NHR_vs_HNR','Harmonicity_NHR','Harmonicity_HNR','Pitch_median','Pitch_mean','Pitch_SD','Pitch_min','Pitch_max','Pulse_no_pulses','Pulse_no_periods','Pulse_mean','Pulse_SD','Voice_fuf','Voice_no_breaks','Voice_deg_Breaks','UPDRS','PD_indicator']

        #Reads data files and adds columns to data file
        dftwosample = pd.read_csv('po1_data.txt', names=column_names)

        #Seperates data files into seperate files if the subject has Parkinsons (PD Indicator of 1) or has no Parkinsons (PD Indicator of 0)
        withparkinsonstwosample = dftwosample[dftwosample["PD_indicator"]==1]
        withoutparktwosample = dftwosample[dftwosample["PD_indicator"]==0]

        #Creates lists to place the outputs into either acceptance or rejection of the null hypothesis
        accept = []
        reject = []

        #Function to compare whats normal for people with Parkinsons to people without Parkinsons
        def sampletest(a, b, c):
            withpark = a
            nopark= b


            x_bar = st.tmean(withpark)
            s = st.tstd(withpark)
            n = len(withpark)


            x_bar1 = st.tmean(nopark)
            s1 = st.tstd(nopark)
            n1 = len(nopark)


            t_stats, p_val = st.ttest_ind_from_stats(x_bar, s, n, x_bar1, s1, n1, equal_var=False, alternative='two-sided')


            print(c, "\t t-statistic (t*): %.9f" % t_stats)


            print(c, "\t p-value: %.9f" % p_val)


            print("\n For", c, ":")
            if p_val < 0.05:
                print("\t We reject the null hypothesis. \n")
                reject.append(c)
            else:
                print("\t We accept the null hypothesis. \n")
                accept.append(c)

        def noprint(a, b, c):
            withpark = a
            nopark= b


            x_bar = st.tmean(withpark)
            s = st.tstd(withpark)
            n = len(withpark)


            x_bar1 = st.tmean(nopark)
            s1 = st.tstd(nopark)
            n1 = len(nopark)


            t_stats, p_val = st.ttest_ind_from_stats(x_bar, s, n, x_bar1, s1, n1, equal_var=False, alternative='two-sided')


            if p_val < 0.05:
                reject.append(c)
            else:
                accept.append(c)

        #Runs the function with each column (excluding the 'Subject Identifier', 'UPDRS' and 'PD indicator' columns)
        noprint(withoutparktwosample.Jitter_percent.values, withparkinsonstwosample.Jitter_percent.values, 'Jitter in %')
        noprint(withoutparktwosample.Jitter_microseconds.values, withparkinsonstwosample.Jitter_microseconds.values, 'Absolute jitter in microseconds' )
        noprint(withoutparktwosample.Jitter_rap.values, withparkinsonstwosample.Jitter_rap.values, 'Jitter as relative amplitude perturbation (r.a.p.)' )
        noprint(withoutparktwosample.Jitter_ppq5.values, withparkinsonstwosample.Jitter_ppq5.values, 'Jitter as 5-point period perturbation quotient (p.p.q.5)' )
        noprint(withoutparktwosample.Jitter_ddp.values, withparkinsonstwosample.Jitter_ddp.values, 'Jitter as average absolute difference of differences between jitter noprints (d.d.p.)' )
        noprint(withoutparktwosample.Shimmer_percent.values, withparkinsonstwosample.Shimmer_percent.values, 'Shimmer in %' )
        noprint(withoutparktwosample.Shimmer_db.values, withparkinsonstwosample.Shimmer_db.values, 'Absolute shimmer in decibels (dB)' )
        noprint(withoutparktwosample.Shimmer_apq3.values, withparkinsonstwosample.Shimmer_apq3.values, 'Shimmer as 3-point amplitude perturbation quotient (a.p.q.3)' )
        noprint(withoutparktwosample.Shimmer_apq5.values, withparkinsonstwosample.Shimmer_apq5.values, 'Shimmer as 5-point amplitude perturbation quotient (a.p.q.5)' )
        noprint(withoutparktwosample.Shimmer_apq11.values, withparkinsonstwosample.Shimmer_apq11.values, 'Shimmer as 11-point amplitude perturbation quotient (a.p.q.11)')
        noprint(withoutparktwosample.Shimmer_dda.values, withparkinsonstwosample.Shimmer_dda.values, 'Shimmer as average absolute differences between consecutive differences between the amplitudes of shimmer noprints (d.d.a.)' )
        noprint(withoutparktwosample.Harmonicity_NHR_vs_HNR.values, withparkinsonstwosample.Harmonicity_NHR_vs_HNR.values, 'Autocorrelation between NHR and HNR' )
        noprint(withoutparktwosample.Harmonicity_NHR.values, withparkinsonstwosample.Harmonicity_NHR.values, 'Noise-to-Harmonic ratio (NHR)' )
        noprint(withoutparktwosample.Harmonicity_HNR.values, withparkinsonstwosample.Harmonicity_HNR.values, 'Harmonic-to-Noise ratio (HNR)' )
        noprint(withoutparktwosample.Pitch_median.values, withparkinsonstwosample.Pitch_median.values, 'Median pitch' )
        noprint(withoutparktwosample.Pitch_mean.values, withparkinsonstwosample.Pitch_mean.values, 'Mean pitch' )
        noprint(withoutparktwosample.Pitch_SD.values, withparkinsonstwosample.Pitch_SD.values, 'Standard deviation of pitch' )
        noprint(withoutparktwosample.Pitch_min.values, withparkinsonstwosample.Pitch_min.values, 'Minimum pitch' )
        noprint(withoutparktwosample.Pitch_max.values, withparkinsonstwosample.Pitch_max.values, 'Maximum pitch' )
        noprint(withoutparktwosample.Pulse_no_pulses.values, withparkinsonstwosample.Pulse_no_pulses.values, 'Number of pulses' )
        noprint(withoutparktwosample.Pulse_no_periods.values, withparkinsonstwosample.Pulse_no_periods.values, 'Number of periods' )
        noprint(withoutparktwosample.Pulse_mean.values, withparkinsonstwosample.Pulse_mean.values, 'Mean period' )
        noprint(withoutparktwosample.Pulse_SD.values, withparkinsonstwosample.Pulse_SD.values, 'Standard deviation of period' )
        noprint(withoutparktwosample.Voice_fuf.values, withparkinsonstwosample.Voice_fuf.values, 'Fraction of unvoiced frames' )
        noprint(withoutparktwosample.Voice_no_breaks.values, withparkinsonstwosample.Voice_no_breaks.values, 'Number of voice breaks' )
        noprint(withoutparktwosample.Voice_deg_Breaks.values, withparkinsonstwosample.Voice_deg_Breaks.values, 'Degree of voice breaks' )

        print('\n')

        #Function to display the accept and reject results in a single column
        def hypothesis(a,b):
            print(a)


            cols = 1
            len1 = len(b)
            rows = int(len1/cols)


            for i in range(rows):
                sublist = b[i*cols]
                print(sublist, )


            print('\n')

        #Runs the 
        hypothesis("################################################################ \n\nDatasets that accept the null hypothesis:", accept)
        hypothesis("Datasets that reject the null hypothesis:", reject)


        print('################################################################\n')
        samplechoice = 1
        while samplechoice != 0:
            print ("If you would like to view the individial column results\n\n", 
                'Select 1 for: Jitter in % \n',
                'Select 2 for: Absolute jitter in microseconds \n',
                'Select 3 for: Jitter as relative amplitude perturbation (r.a.p.) \n',
                'Select 4 for: Jitter as 5-point period perturbation quotient (p.p.q.5) \n',
                'Select 5 for: Jitter as average absolute difference of differences between jitter confidences (d.d.p.) \n',
                'Select 6 for: Shimmer in % \n',
                'Select 7 for: Absolute shimmer in decibels (dB) \n',
                'Select 8 for: Shimmer as 3-point amplitude perturbation quotient (a.p.q.3) \n',
                'Select 9 for: Shimmer as 5-point amplitude perturbation quotient (a.p.q.5)\n',
                'Select 10 for: Shimmer as 11-point amplitude perturbation quotient (a.p.q.11) \n',
                'Select 11 for: Shimmer as average absolute differences between consecutive differences between the amplitudes of shimmer confidences (d.d.a.) \n',
                'Select 12 for: Autocorrelation between NHR and HNR \n',
                'Select 13 for: Noise-to-Harmonic ratio (NHR) \n',
                'Select 14 for: Harmonic-to-Noise ratio (HNR) \n',
                'Select 15 for: Median pitch \n',
                'Select 16 for: Mean pitch \n',
                'Select 17 for: Standard deviation of pitch \n',
                'Select 18 for: Minimum Pitch \n',
                'Select 19 for: Maximum Pitch \n',
                'Select 20 for: Number of Pulses \n',
                'Select 21 for: Numer of periods \n',
                'Select 22 for: Mean Pulse \n',
                'Select 23 for: Standard deviation of period \n',
                'Select 24 for: Fraction of unvoiced frames \n',
                'Select 25 for: Number of voice breaks \n',
                'Select 26 for: Degree of voice breaks \n \n',
                'Or select 0 to return to the Main Menu \n')
            
            
            samplechoice = input("Selection: ")

            if int(samplechoice) == 1:
                sampletest(withoutparktwosample.Jitter_percent.values, withparkinsonstwosample.Jitter_percent.values, 'Jitter in %')

            elif int(samplechoice) == 2:
                sampletest(withoutparktwosample.Jitter_microseconds.values, withparkinsonstwosample.Jitter_microseconds.values, 'Absolute jitter in microseconds' )

            elif int(samplechoice) == 3:
                sampletest(withoutparktwosample.Jitter_rap.values, withparkinsonstwosample.Jitter_rap.values, 'Jitter as relative amplitude perturbation (r.a.p.)' )

            elif int(samplechoice) == 4:
                sampletest(withoutparktwosample.Jitter_ppq5.values, withparkinsonstwosample.Jitter_ppq5.values, 'Jitter as 5-point period perturbation quotient (p.p.q.5)' )
            
            elif int(samplechoice) == 5:
                sampletest(withoutparktwosample.Jitter_ddp.values, withparkinsonstwosample.Jitter_ddp.values, 'Jitter as average absolute difference of differences between jitter sampletests (d.d.p.)' )

            elif int(samplechoice) == 6:
                sampletest(withoutparktwosample.Shimmer_percent.values, withparkinsonstwosample.Shimmer_percent.values, 'Shimmer in %' )

            elif int(samplechoice) == 7:
                sampletest(withoutparktwosample.Shimmer_db.values, withparkinsonstwosample.Shimmer_db.values, 'Absolute shimmer in decibels (dB)' )

            elif int(samplechoice) == 8:
                sampletest(withoutparktwosample.Shimmer_apq3.values, withparkinsonstwosample.Shimmer_apq3.values, 'Shimmer as 3-point amplitude perturbation quotient (a.p.q.3)' )

            elif int(samplechoice) == 9:
                sampletest(withoutparktwosample.Shimmer_apq5.values, withparkinsonstwosample.Shimmer_apq5.values, 'Shimmer as 5-point amplitude perturbation quotient (a.p.q.5)' )

            elif int(samplechoice) == 10:
                sampletest(withoutparktwosample.Shimmer_apq11.values, withparkinsonstwosample.Shimmer_apq11.values, 'Shimmer as 11-point amplitude perturbation quotient (a.p.q.11)')

            elif int(samplechoice) == 11:
                sampletest(withoutparktwosample.Shimmer_dda.values, withparkinsonstwosample.Shimmer_dda.values, 'Shimmer as average absolute differences between consecutive differences between the amplitudes of shimmer sampletests (d.d.a.)' )

            elif int(samplechoice) == 12:
                sampletest(withoutparktwosample.Harmonicity_NHR_vs_HNR.values, withparkinsonstwosample.Harmonicity_NHR_vs_HNR.values, 'Autocorrelation between NHR and HNR' )

            elif int(samplechoice) == 13:
                sampletest(withoutparktwosample.Harmonicity_NHR.values, withparkinsonstwosample.Harmonicity_NHR.values, 'Noise-to-Harmonic ratio (NHR)' )

            elif int(samplechoice) == 14:
                sampletest(withoutparktwosample.Harmonicity_HNR.values, withparkinsonstwosample.Harmonicity_HNR.values, 'Harmonic-to-Noise ratio (HNR)' )

            elif int(samplechoice) == 15:
                sampletest(withoutparktwosample.Pitch_median.values, withparkinsonstwosample.Pitch_median.values, 'Median pitch' )

            elif int(samplechoice) == 16:
                sampletest(withoutparktwosample.Pitch_mean.values, withparkinsonstwosample.Pitch_mean.values, 'Mean pitch' )

            elif int(samplechoice) == 17:
                sampletest(withoutparktwosample.Pitch_SD.values, withparkinsonstwosample.Pitch_SD.values, 'Standard deviation of pitch' )

            elif int(samplechoice) == 18:
                sampletest(withoutparktwosample.Pitch_min.values, withparkinsonstwosample.Pitch_min.values, 'Minimum pitch' )

            elif int(samplechoice) == 19:
                sampletest(withoutparktwosample.Pitch_max.values, withparkinsonstwosample.Pitch_max.values, 'Maximum pitch' )

            elif int(samplechoice) == 20:
                sampletest(withoutparktwosample.Pulse_no_pulses.values, withparkinsonstwosample.Pulse_no_pulses.values, 'Number of pulses' )

            elif int(samplechoice) == 21:
                sampletest(withoutparktwosample.Pulse_no_periods.values, withparkinsonstwosample.Pulse_no_periods.values, 'Number of periods' )

            elif int(samplechoice) == 22:
                sampletest(withoutparktwosample.Pulse_mean.values, withparkinsonstwosample.Pulse_mean.values, 'Mean period' )

            elif int(samplechoice) == 23:
                sampletest(withoutparktwosample.Pulse_SD.values, withparkinsonstwosample.Pulse_SD.values, 'Standard deviation of period' )

            elif int(samplechoice) == 24:
                sampletest(withoutparktwosample.Voice_fuf.values, withparkinsonstwosample.Voice_fuf.values, 'Fraction of unvoiced frames' )

            elif int(samplechoice) == 25:
                sampletest(withoutparktwosample.Voice_no_breaks.values, withparkinsonstwosample.Voice_no_breaks.values, 'Number of voice breaks' )

            elif int(samplechoice) == 26:
                sampletest(withoutparktwosample.Voice_deg_Breaks.values, withparkinsonstwosample.Voice_deg_Breaks.values, 'Degree of voice breaks' )  

            else:
                print("Main Menu \n")
                break

##################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
    elif int(userchoice) == 5:
        
        # Column names to be added
        column_names=['Subject_identifier','Jitter_percent','Jitter_microseconds','Jitter_rap','Jitter_ppq5','Jitter_ddp','Shimmer_percent','Shimmer_db','Shimmer_apq3','Shimmer_apq5','Shimmer_apq11','Shimmer_dda','Harmonicity_NHR_vs_HNR','Harmonicity_NHR','Harmonicity_HNR','Pitch_median','Pitch_mean','Pitch_SD','Pitch_min','Pitch_max','Pulse_no_pulses','Pulse_no_periods','Pulse_mean','Pulse_SD','Voice_fuf','Voice_no_breaks','Voice_deg_Breaks','UPDRS','PD_indicator']

        #Reads data files and adds columns to data file
        dfcentral = pd.read_csv('po1_data.txt', names=column_names)

        #Seperates data files into seperate files if the subject has Parkinsons (PD Indicator of 1) or has no Parkinsons (PD Indicator of 0)
        withparkinsonscentral = dfcentral[dfcentral["PD_indicator"]==1]
        withoutparkcentral = dfcentral[dfcentral["PD_indicator"]==0]



        #Function to calculate the mean, mode and median of people with Parkinsons vs without Parkinsons
        def central(a, b, c):
            #Line break to assist with review of information
            print('--------------------------------------------------------------------')
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
            print('Without Parkinsons data:','''
            ''',c, 'mean:', woparkmean,'''
            ''',c, 'mode:', woparkmode, '''
            ''',c, 'median:', woparkmedian,'''
            ''''\n','With Parkinsons:','''
            ''',c, 'mean:', wparkmean, '''
            ''',c, 'mode:', wparkmode, '''
            ''',c, 'median:', wparkmedian,'''
            ''' )

            #Line break to assist with review of information
            print('--------------------------------------------------------------------')

        centraltendancy = 1
        while centraltendancy != 0:
            print ("To view the individial columns central tendancy make a selection\n\n", 
                'Select 1 for: Jitter in % \n',
                'Select 2 for: Absolute jitter in microseconds \n',
                'Select 3 for: Jitter as relative amplitude perturbation (r.a.p.) \n',
                'Select 4 for: Jitter as 5-point period perturbation quotient (p.p.q.5) \n',
                'Select 5 for: Jitter as average absolute difference of differences between jitter confidences (d.d.p.) \n',
                'Select 6 for: Shimmer in % \n',
                'Select 7 for: Absolute shimmer in decibels (dB) \n',
                'Select 8 for: Shimmer as 3-point amplitude perturbation quotient (a.p.q.3) \n',
                'Select 9 for: Shimmer as 5-point amplitude perturbation quotient (a.p.q.5)\n',
                'Select 10 for: Shimmer as 11-point amplitude perturbation quotient (a.p.q.11) \n',
                'Select 11 for: Shimmer as average absolute differences between consecutive differences between the amplitudes of shimmer confidences (d.d.a.) \n',
                'Select 12 for: Autocorrelation between NHR and HNR \n',
                'Select 13 for: Noise-to-Harmonic ratio (NHR) \n',
                'Select 14 for: Harmonic-to-Noise ratio (HNR) \n',
                'Select 15 for: Median pitch \n',
                'Select 16 for: Mean pitch \n',
                'Select 17 for: Standard deviation of pitch \n',
                'Select 18 for: Minimum Pitch \n',
                'Select 19 for: Maximum Pitch \n',
                'Select 20 for: Number of Pulses \n',
                'Select 21 for: Numer of periods \n',
                'Select 22 for: Mean Pulse \n',
                'Select 23 for: Standard deviation of period \n',
                'Select 24 for: Fraction of unvoiced frames \n',
                'Select 25 for: Number of voice breaks \n',
                'Select 26 for: Degree of voice breaks \n \n',
                'Or select 0 to return to the Main Menu \n\n')
            
            
            centraltendancy = input("Selection: ")

            if int(centraltendancy) == 1:
                central(withoutparkcentral.Jitter_percent.values, withparkinsonscentral.Jitter_percent.values, 'Jitter in %')

            elif int(centraltendancy) == 2:
                central(withoutparkcentral.Jitter_microseconds.values, withparkinsonscentral.Jitter_microseconds.values, 'Absolute jitter in microseconds' )


            elif int(centraltendancy) == 3:
                central(withoutparkcentral.Jitter_rap.values, withparkinsonscentral.Jitter_rap.values, 'Jitter as relative amplitude perturbation (r.a.p.)' )

            elif int(centraltendancy) == 4:      
                central(withoutparkcentral.Jitter_ppq5.values, withparkinsonscentral.Jitter_ppq5.values, 'Jitter as 5-point period perturbation quotient (p.p.q.5)' )
            
            elif int(centraltendancy) == 5:
                central(withoutparkcentral.Jitter_ddp.values, withparkinsonscentral.Jitter_ddp.values, 'Jitter as average absolute difference of differences between jitter centrals (d.d.p.)' )

            elif int(centraltendancy) == 6:
                central(withoutparkcentral.Shimmer_percent.values, withparkinsonscentral.Shimmer_percent.values, 'Shimmer in %' )

            elif int(centraltendancy) == 7:
                central(withoutparkcentral.Shimmer_db.values, withparkinsonscentral.Shimmer_db.values, 'Absolute shimmer in decibels (dB)' )

            elif int(centraltendancy) == 8:
                central(withoutparkcentral.Shimmer_apq3.values, withparkinsonscentral.Shimmer_apq3.values, 'Shimmer as 3-point amplitude perturbation quotient (a.p.q.3)' )

            elif int(centraltendancy) == 9:
                central(withoutparkcentral.Shimmer_apq5.values, withparkinsonscentral.Shimmer_apq5.values, 'Shimmer as 5-point amplitude perturbation quotient (a.p.q.5)' )

            elif int(centraltendancy) == 10:
                central(withoutparkcentral.Shimmer_apq11.values, withparkinsonscentral.Shimmer_apq11.values, 'Shimmer as 11-point amplitude perturbation quotient (a.p.q.11)')

            elif int(centraltendancy) == 11:
                central(withoutparkcentral.Shimmer_dda.values, withparkinsonscentral.Shimmer_dda.values, 'Shimmer as average absolute differences between consecutive differences between the amplitudes of shimmer centrals (d.d.a.)' )

            elif int(centraltendancy) == 12:
                central(withoutparkcentral.Harmonicity_NHR_vs_HNR.values, withparkinsonscentral.Harmonicity_NHR_vs_HNR.values, 'Autocorrelation between NHR and HNR' )

            elif int(centraltendancy) == 13:
                central(withoutparkcentral.Harmonicity_NHR.values, withparkinsonscentral.Harmonicity_NHR.values, 'Noise-to-Harmonic ratio (NHR)' )

            elif int(centraltendancy) == 14:
                central(withoutparkcentral.Harmonicity_HNR.values, withparkinsonscentral.Harmonicity_HNR.values, 'Harmonic-to-Noise ratio (HNR)' )

            elif int(centraltendancy) == 15:
                central(withoutparkcentral.Pitch_median.values, withparkinsonscentral.Pitch_median.values, 'Median pitch' )

            elif int(centraltendancy) == 16:
                central(withoutparkcentral.Pitch_mean.values, withparkinsonscentral.Pitch_mean.values, 'Mean pitch' )

            elif int(centraltendancy) == 17:
                central(withoutparkcentral.Pitch_SD.values, withparkinsonscentral.Pitch_SD.values, 'Standard deviation of pitch' )

            elif int(centraltendancy) == 18:
                central(withoutparkcentral.Pitch_min.values, withparkinsonscentral.Pitch_min.values, 'Minimum pitch' )

            elif int(centraltendancy) == 19:
                central(withoutparkcentral.Pitch_max.values, withparkinsonscentral.Pitch_max.values, 'Maximum pitch' )

            elif int(centraltendancy) == 20:
                central(withoutparkcentral.Pulse_no_pulses.values, withparkinsonscentral.Pulse_no_pulses.values, 'Number of pulses' )

            elif int(centraltendancy) == 21:
                central(withoutparkcentral.Pulse_no_periods.values, withparkinsonscentral.Pulse_no_periods.values, 'Number of periods' )

            elif int(centraltendancy) == 22:
                central(withoutparkcentral.Pulse_mean.values, withparkinsonscentral.Pulse_mean.values, 'Mean period' )

            elif int(centraltendancy) == 23:
                central(withoutparkcentral.Pulse_SD.values, withparkinsonscentral.Pulse_SD.values, 'Standard deviation of period' )

            elif int(centraltendancy) == 24:
                central(withoutparkcentral.Voice_fuf.values, withparkinsonscentral.Voice_fuf.values, 'Fraction of unvoiced frames' )

            elif int(centraltendancy) == 25:
                central(withoutparkcentral.Voice_no_breaks.values, withparkinsonscentral.Voice_no_breaks.values, 'Number of voice breaks' )

            elif int(centraltendancy) == 26:
                central(withoutparkcentral.Voice_deg_Breaks.values, withparkinsonscentral.Voice_deg_Breaks.values, 'Degree of voice breaks' )    

            else:
                print("Main Menu\n")
                break
   
##################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################    
    else:
        print("\nGoodbye!\n")
        break