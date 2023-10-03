import pandas as pd
from sklearn import linear_model
import tkinter as tk 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import scipy.stats as st
from tkinter import *   
import seaborn as sns
from tkinter import messagebox  
from tkinter import ttk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import math
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

#Reads data files and adds columns to data file
df = pd.read_csv('po2_data.csv')
column_names = ['subject#','age','sex','test_time','motor_updrs','total_updrs','jitter_percent', 'jitter_abs', 'jitter_rap', 'jitter_ppq5', 'jitter_ddp', 'shimmer_percent', 'shimmer_abs', 'shimmer_apq3', 'shimmer_apq5', 'shimmer_apq11', 'shimmer_dda', 'nhr', 'hnr', 'rpde', 'dfa', 'ppe']
df.columns= column_names
male = df[df["sex"]==0]
female = df[df["sex"]==1]

root= tk.Tk()
root.geometry('780x900')

mainframe = Frame(root)
mainframe.pack(fill=BOTH, expand=1)

mycanvas = Canvas(mainframe)
mycanvas.pack(side=LEFT, fill=BOTH, expand=1)

scrollbar = tk.Scrollbar(mainframe, orient=VERTICAL, command=mycanvas.yview)
scrollbar.pack(side=RIGHT, fill=Y)
scrollbar.bind_all('<MouseWheel>')


def _on_mousewheel(event):
    mycanvas.yview_scroll(int(-1*(event.delta/120)), "units")

mycanvas.configure(yscrollcommand=scrollbar.set)
mycanvas.bind('<Configure>', lambda e:mycanvas.configure(scrollregion=mycanvas.bbox("all")))
root.bind_all('<MouseWheel>', _on_mousewheel)

secondcanvasframe = Frame(mycanvas)
mycanvas.create_window((0,0), window=secondcanvasframe, anchor='nw')

titleframe=Frame(secondcanvasframe, highlightbackground='black', highlightthickness=1)
titleframe.pack(pady=1)

topframe = Frame(secondcanvasframe, highlightbackground='black', highlightthickness=1)
topframe.pack(pady=1)

textboxframetitle = Frame(secondcanvasframe)
textboxframetitle.pack(pady=3)

textboxframe = Frame(secondcanvasframe, highlightbackground='black', highlightthickness=1)
textboxframe.pack(pady=1)

inputsectionframe = Frame(secondcanvasframe, highlightbackground='black', highlightthickness=1)
inputsectionframe.pack(pady=1)

malefemalesectionframe = Frame(secondcanvasframe, highlightbackground='black', highlightthickness=1)
malefemalesectionframe.pack(pady=1)

maleselect = Frame(malefemalesectionframe, highlightbackground='black', highlightthickness=1)
maleselect.pack(pady=1,side=LEFT)

femaleselect = Frame(malefemalesectionframe, highlightbackground='black', highlightthickness=1)
femaleselect.pack(pady=1,side=RIGHT)

subjectselectionframe = Frame(secondcanvasframe, highlightbackground='black', highlightthickness=1)
subjectselectionframe.pack(pady=1)

linregress = Frame(secondcanvasframe, highlightbackground='black', highlightthickness=1)
linregress.pack(pady=1)
#################################################################################################################
multilinregresstitle= Frame(secondcanvasframe, highlightbackground='black', highlightthickness=1)
multilinregresstitle.pack(pady=1)

multilinregressmore = Frame(secondcanvasframe, highlightbackground='black', highlightthickness=1)
multilinregressmore.pack(pady=1)

multilinregressbutton = Frame(secondcanvasframe, highlightbackground='black', highlightthickness=1)
multilinregressbutton.pack(pady=1)

multilinregressrescaletitle = Frame(secondcanvasframe, highlightbackground='black', highlightthickness=1)
multilinregressrescaletitle.pack(pady=1)

multilinregressrescale = Frame(secondcanvasframe, highlightbackground='black', highlightthickness=1)
multilinregressrescale.pack(pady=1)

multilinregressrescalebutton = Frame(secondcanvasframe, highlightbackground='black', highlightthickness=1)
multilinregressrescalebutton.pack(pady=1)
############################################################################################################################

linregressbysubject = Frame(secondcanvasframe, highlightbackground='black', highlightthickness=1)
linregressbysubject.pack(pady=1)


histoframe1 = Frame(secondcanvasframe, highlightbackground='black', highlightthickness=1)
histoframe1.pack(pady=1)

histoframe2 = Frame(secondcanvasframe, highlightbackground='black', highlightthickness=1)
histoframe2.pack(pady=1)




footerframe= Frame(secondcanvasframe)
footerframe.pack(pady=1)

title = Label(titleframe, text = "HIT140 Group 6")
title.config(font =("ariel", 20))
title.pack(pady=1)


groupmembers = Label(titleframe, text = "Mark Connelly ** Paulino Jouth ** Guy Watton ** Nicole Wilson")
groupmembers.config(font =("ariel", 11))
groupmembers.pack(pady=1)

###############################################################################################################################################



def simpcorrel():
    corr = df.corr()

    # Plot the pairwise correlation as heatmap
    ax = sns.heatmap(
        corr, 
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=False,
        annot=True
    )

    # customise the labels
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )

    plt.show()


simpcorlab = Label(topframe, text = "Generate Overall \nCorrelation Matrix")
simpcorlab.config(font =("ariel", 10))
simpcorlab.pack()
buttoncorr = tk.Button (topframe, text='Generate',command=simpcorrel, bg='light blue', width=10, height=5)
buttoncorr.pack()





###############################################################################################################################################
#Correlation Matrix Male Vs Female

genmale = male[['age','test_time','motor_updrs','total_updrs','jitter_percent', 'jitter_abs', 'jitter_rap', 'jitter_ppq5', 'jitter_ddp', 'shimmer_percent', 'shimmer_abs', 'shimmer_apq3', 'shimmer_apq5', 'shimmer_apq11', 'shimmer_dda', 'nhr', 'hnr', 'rpde', 'dfa', 'ppe']]
genfemale = female[['age','test_time','motor_updrs','total_updrs','jitter_percent', 'jitter_abs', 'jitter_rap', 'jitter_ppq5', 'jitter_ddp', 'shimmer_percent', 'shimmer_abs', 'shimmer_apq3', 'shimmer_apq5', 'shimmer_apq11', 'shimmer_dda', 'nhr', 'hnr', 'rpde', 'dfa', 'ppe']]


def simpcorrelmale():
    corr = genmale.corr()

    # Plot the pairwise correlation as heatmap
    ax = sns.heatmap(
        corr, 
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=False,
        annot=True
    )

    # customise the labels
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )

    plt.show()


simpcorlabmale = Label(topframe, text = "Generate Male Only \nCorrelation Matrix")
simpcorlabmale.config(font =("ariel", 10))
simpcorlabmale.pack()
buttoncorrmale = tk.Button (topframe, text='Generate',command=simpcorrelmale, bg='light blue', width=10, height=5)
buttoncorrmale.pack()


def simpcorrelfemale():
    corr = genfemale.corr()

    # Plot the pairwise correlation as heatmap
    ax = sns.heatmap(
        corr, 
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=False,
        annot=True
    )

    # customise the labels
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )

    plt.show()


simpcorlabfemale = Label(topframe, text = "Generate Female Only\nCorrelation Matrix")
simpcorlabfemale.config(font =("ariel", 10))
simpcorlabfemale.pack()
buttoncorrfemale = tk.Button (topframe, text='Generate',command=simpcorrelfemale, bg='light blue', width=10, height=5)
buttoncorrfemale.pack()

###############################################################################################################################################
def simpcorrelgroup1():
    corr = df.corr()

    # Plot the pairwise correlation as heatmap
    ax = sns.heatmap(
        corr, 
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=False,
        annot=True
    )

    # customise the labels
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )

    plt.show()


simpcorlab = Label(topframe, text = "Generate Overall \nCorrelation Matrix")
simpcorlab.config(font =("ariel", 10))
simpcorlab.pack()
buttoncorr = tk.Button (topframe, text='Generate',command=simpcorrel, bg='light blue', width=10, height=5)
buttoncorr.pack()



###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################

#Column Data
columndatalabel = Label(textboxframetitle, text = 'Column Description Data')
columndatalabel.config(font =("ariel", 18))
columndatalabel.pack()


def jitter_percentgetdetails():
    fklist = []
    fklist.append(df["jitter_percent"].describe())
    messagebox.showinfo('jitter_percent', fklist )
def jitter_absgetdetails():
    fklist = []
    fklist.append(df["jitter_abs"].describe())
    messagebox.showinfo('jitter_abs', fklist )
def jitter_rapgetdetails():
    fklist = []
    fklist.append(df["jitter_rap"].describe())
    messagebox.showinfo('jitter_rap', fklist )
def jitter_ppq5getdetails():
    fklist = []
    fklist.append(df["jitter_ppq5"].describe())
    messagebox.showinfo('jitter_ppq5', fklist )
def jitter_ddpgetdetails():
    fklist = []
    fklist.append(df["jitter_ddp"].describe())
    messagebox.showinfo('jitter_ddp', fklist )
def shimmer_percentgetdetails():
    fklist = []
    fklist.append(df["shimmer_percent"].describe())
    messagebox.showinfo('shimmer_percent', fklist )
def shimmer_absgetdetails():
    fklist = []
    fklist.append(df["shimmer_abs"].describe())
    messagebox.showinfo('shimmer_abs', fklist )
def shimmer_apq3getdetails():
    fklist = []
    fklist.append(df["shimmer_apq3"].describe())
    messagebox.showinfo('shimmer_apq3', fklist )
def shimmer_apq5getdetails():
    fklist = []
    fklist.append(df["shimmer_apq5"].describe())
    messagebox.showinfo('shimmer_apq5', fklist )
def shimmer_apq11getdetails():
    fklist = []
    fklist.append(df["shimmer_apq11"].describe())
    messagebox.showinfo('shimmer_apq11', fklist )
def shimmer_ddagetdetails():
    fklist = []
    fklist.append(df["shimmer_dda"].describe())
    messagebox.showinfo('shimmer_dda', fklist )
def nhrgetdetails():
    fklist = []
    fklist.append(df["nhr"].describe())
    messagebox.showinfo('nhr', fklist )
def hnrgetdetails():
    fklist = []
    fklist.append(df["hnr"].describe())
    messagebox.showinfo('hnr', fklist )
def rpdegetdetails():
    fklist = []
    fklist.append(df["rpde"].describe())
    messagebox.showinfo('rpde', fklist )
def dfagetdetails():
    fklist = []
    fklist.append(df["dfa"].describe())
    messagebox.showinfo('dfa', fklist )
def ppegetdetails():
    fklist = []
    fklist.append(df["ppe"].describe())
    messagebox.showinfo('ppe', fklist )

def summarybuttoncreate(a,b,c,d,e):
    a = tk.Button (textboxframe, text=b,command= c, bg='light blue', height=1, width=15)
    a.grid(row=d,column=e,padx=2,pady=2)

# jitterperdetsbutton = tk.Button (textboxframe, text='Generate',command= getdetails, bg='light blue')
# jitterperdetsbutton.pack(pady=20)

summarybuttoncreate('jitterperdetsbutton', 'Jitter(%)', jitter_percentgetdetails,1,1)
summarybuttoncreate('jitter_absdetsbutton', 'jitterabs',jitter_absgetdetails,1,2)
summarybuttoncreate('jitter_rapdetsbutton','jitter_rap', jitter_rapgetdetails,1,3)
summarybuttoncreate('jitter_ppq5detsbutton','jitter_ppq5',jitter_ppq5getdetails,1,4)
summarybuttoncreate('jitter_ddpdetsbutton','jitter_ddp',jitter_ddpgetdetails,1,5)
summarybuttoncreate('shimmer_percentdetsbutton','shimmer_percent',shimmer_percentgetdetails,2,1)
summarybuttoncreate('shimmer_absdetsbutton','shimmer_abs',shimmer_absgetdetails,2,2)
summarybuttoncreate('shimmer_apq3detsbutton','shimmer_apq3',shimmer_apq3getdetails,2,3)
summarybuttoncreate('shimmer_apq5detsbutton','shimmer_apq5',shimmer_apq5getdetails,2,4)
summarybuttoncreate('shimmer_apq11detsbutton','shimmer_apq11',shimmer_apq11getdetails,2,5)
summarybuttoncreate('shimmer_ddadetsbutton','shimmer_dda',shimmer_ddagetdetails,3,1)
summarybuttoncreate('nhrdetsbutton','nhr',nhrgetdetails,3,2)
summarybuttoncreate('hnrdetsbutton','hnr',hnrgetdetails,3,3)
summarybuttoncreate('rpdedetsbutton','rpde',rpdegetdetails,3,4)
summarybuttoncreate('dfadetsbutton','dfa',dfagetdetails,3,5)
summarybuttoncreate('ppedetsbutton','ppe',ppegetdetails,4,1)


###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################
#Scatterplots Both Genders
selectionlist = Label(inputsectionframe, text = "Scatterplot with Regression Line\nSelect Two Columns To Compare")
selectionlist.config(font =("ariel", 18))
selectionlist.pack(pady=5)

dropdown1label = Label(inputsectionframe, text = "X-Axis")
dropdown1label.config(font =("ariel", 11))
dropdown1label.pack(pady=5)
dropdown1 = ttk.Combobox(inputsectionframe,state="readonly",values=['age','sex','test_time','motor_updrs','total_updrs','jitter_percent', 'jitter_abs', 'jitter_rap', 'jitter_ppq5', 'jitter_ddp', 'shimmer_percent', 'shimmer_abs', 'shimmer_apq3', 'shimmer_apq5', 'shimmer_apq11', 'shimmer_dda', 'nhr', 'hnr', 'rpde', 'dfa', 'ppe'])
dropdown1.pack(pady=5)


dropdown2label = Label(inputsectionframe, text = "Y-Axis")
dropdown2label.config(font =("ariel", 11))
dropdown2label.pack(pady=5)
dropdown2 = ttk.Combobox(inputsectionframe,state="readonly",values=['age','sex','test_time','motor_updrs','total_updrs','jitter_percent', 'jitter_abs', 'jitter_rap', 'jitter_ppq5', 'jitter_ddp', 'shimmer_percent', 'shimmer_abs', 'shimmer_apq3', 'shimmer_apq5', 'shimmer_apq11', 'shimmer_dda', 'nhr', 'hnr', 'rpde', 'dfa', 'ppe'])
dropdown2.pack(pady=5)

def selectionlists():
    var3 = df[dropdown1.get()]
    var4 = df[dropdown2.get()]
    plt.plot(var3, var4, 'o')
    m, b = np.polyfit(var3, var4, 1)
    plt.plot(var3, m*var3+b)
    plt.scatter(var3, var4)
    plt.title('Scatterplot')
    plt.xlabel(str(dropdown1.get()))
    plt.ylabel(str(dropdown2.get()))
    plt.show()



selectionlistbutton= tk.Button (inputsectionframe, text='Generate',command=selectionlists, bg='light blue')
selectionlistbutton.pack()
###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################
#Scatterplots Male and Female
#Male
selectionlistmale = Label(maleselect, text = "\n** Male Only **\nScatterplot with Regression Line\nSelect Two Columns To Compare")
selectionlistmale.config(font =("ariel", 18))
selectionlistmale.pack(pady=5)

dropdown1malelabel = Label(maleselect, text = "X-Axis")
dropdown1malelabel.config(font =("ariel", 11))
dropdown1malelabel.pack(pady=5)
dropdown1male = ttk.Combobox(maleselect,state="readonly",values=['age','sex','test_time','motor_updrs','total_updrs','jitter_percent', 'jitter_abs', 'jitter_rap', 'jitter_ppq5', 'jitter_ddp', 'shimmer_percent', 'shimmer_abs', 'shimmer_apq3', 'shimmer_apq5', 'shimmer_apq11', 'shimmer_dda', 'nhr', 'hnr', 'rpde', 'dfa', 'ppe'])
dropdown1male.pack(pady=5)


dropdown2malelabel = Label(maleselect, text = "Y-Axis")
dropdown2malelabel.config(font =("ariel", 11))
dropdown2malelabel.pack(pady=5)
dropdown2male = ttk.Combobox(maleselect,state="readonly",values=['age','sex','test_time','motor_updrs','total_updrs','jitter_percent', 'jitter_abs', 'jitter_rap', 'jitter_ppq5', 'jitter_ddp', 'shimmer_percent', 'shimmer_abs', 'shimmer_apq3', 'shimmer_apq5', 'shimmer_apq11', 'shimmer_dda', 'nhr', 'hnr', 'rpde', 'dfa', 'ppe'])
dropdown2male.pack(pady=5)

def selectionlistsmale():
    var3 = male[dropdown1male.get()]
    var4 = male[dropdown2male.get()]
    plt.plot(var3, var4, 'o')
    m, b = np.polyfit(var3, var4, 1)
    plt.plot(var3, m*var3+b)
    plt.scatter(var3, var4)
    plt.title('Scatterplot')
    plt.xlabel(str(dropdown1male.get()))
    plt.ylabel(str(dropdown2male.get()))
    plt.show()


selectionlistbuttonmale= tk.Button (maleselect, text='Generate',command=selectionlistsmale, bg='light blue')
selectionlistbuttonmale.pack(pady=5)

#Female
selectionlistfemale = Label(femaleselect, text = "\n** Female Only **\nScatterplot with Regression Line\nSelect Two Columns To Compare")
selectionlistfemale.config(font =("ariel", 18))
selectionlistfemale.pack(pady=5)

dropdown1femalelabel = Label(femaleselect, text = "X-Axis")
dropdown1femalelabel.config(font =("ariel", 11))
dropdown1femalelabel.pack(pady=5)
dropdown1female = ttk.Combobox(femaleselect,state="readonly",values=['age','sex','test_time','motor_updrs','total_updrs','jitter_percent', 'jitter_abs', 'jitter_rap', 'jitter_ppq5', 'jitter_ddp', 'shimmer_percent', 'shimmer_abs', 'shimmer_apq3', 'shimmer_apq5', 'shimmer_apq11', 'shimmer_dda', 'nhr', 'hnr', 'rpde', 'dfa', 'ppe'])
dropdown1female.pack(pady=5)


dropdown2femalelabel = Label(femaleselect, text = "Y-Axis")
dropdown2femalelabel.config(font =("ariel", 11))
dropdown2femalelabel.pack(pady=5)
dropdown2female = ttk.Combobox(femaleselect,state="readonly",values=['age','sex','test_time','motor_updrs','total_updrs','jitter_percent', 'jitter_abs', 'jitter_rap', 'jitter_ppq5', 'jitter_ddp', 'shimmer_percent', 'shimmer_abs', 'shimmer_apq3', 'shimmer_apq5', 'shimmer_apq11', 'shimmer_dda', 'nhr', 'hnr', 'rpde', 'dfa', 'ppe'])
dropdown2female.pack(pady=5)

def selectionlistsfemale():
    var3 = female[dropdown1female.get()]
    var4 = female[dropdown2female.get()]
    plt.plot(var3, var4, 'o')
    m, b = np.polyfit(var3, var4, 1)
    plt.plot(var3, m*var3+b)
    plt.scatter(var3, var4)
    plt.title('Scatterplot')
    plt.xlabel(str(dropdown1female.get()))
    plt.ylabel(str(dropdown2female.get()))
    plt.show()



selectionlistbuttonfemale= tk.Button (femaleselect, text='Generate',command=selectionlistsfemale, bg='light blue')
selectionlistbuttonfemale.pack(pady=5)

###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################


selectionlistsubject = Label(subjectselectionframe, text = "Scatterplot with Regression Line\nBy Indivual Subject\nSelect Two Columns To Compare")
selectionlistsubject.config(font =("ariel", 18))
selectionlistsubject.pack(pady=5)


dropdownsub1label = Label(subjectselectionframe, text = "X-Axis")
dropdownsub1label.config(font =("ariel", 11))
dropdownsub1label.pack(pady=5)
dropdown1bysubject = ttk.Combobox(subjectselectionframe,state="readonly",values=['age','sex','test_time','motor_updrs','total_updrs','jitter_percent', 'jitter_abs', 'jitter_rap', 'jitter_ppq5', 'jitter_ddp', 'shimmer_percent', 'shimmer_abs', 'shimmer_apq3', 'shimmer_apq5', 'shimmer_apq11', 'shimmer_dda', 'nhr', 'hnr', 'rpde', 'dfa', 'ppe'])
dropdown1bysubject.pack(pady=5)


dropdownsub2label = Label(subjectselectionframe, text = "Y-Axis")
dropdownsub2label.config(font =("ariel", 11))
dropdownsub2label.pack(pady=5)
dropdown2bysubject = ttk.Combobox(subjectselectionframe,state="readonly",values=['age','sex','test_time','motor_updrs','total_updrs','jitter_percent', 'jitter_abs', 'jitter_rap', 'jitter_ppq5', 'jitter_ddp', 'shimmer_percent', 'shimmer_abs', 'shimmer_apq3', 'shimmer_apq5', 'shimmer_apq11', 'shimmer_dda', 'nhr', 'hnr', 'rpde', 'dfa', 'ppe'])
dropdown2bysubject.pack(pady=5)


subjectentlabel = Label(subjectselectionframe, text = "Enter Subject Number")
subjectentlabel.config(font =("ariel", 11))
subjectentlabel.pack(pady=5)
subjectent = tk.Entry (subjectselectionframe) 
subjectent.pack(pady=5)


def subjectselectionlists():
    bananas1 = int(subjectent.get())
    subjectnumber1 = df[df['subject#'] == bananas1]
    var3 = subjectnumber1[dropdown1bysubject.get()]
    var4 = subjectnumber1[dropdown2bysubject.get()]
    plt.plot(var3, var4, 'o')
    m, b = np.polyfit(var3, var4, 1)
    plt.plot(var3, m*var3+b)
    plt.scatter(var3, var4)
    plt.title('Scatterplot')
    plt.xlabel(str(dropdown1.get()))
    plt.ylabel(str(dropdown2.get()))
    plt.show()


selectionlistbysubjectbutton= tk.Button (subjectselectionframe, text='Generate',command=subjectselectionlists, bg='light blue')
selectionlistbysubjectbutton.pack()


###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################

#Histogram function to generate a histogram based on the desired column.
def histogram1 ():
    histodata = df[histodropdown1.get()]

    # Calculate the number of bins
    num_bins = int(np.sqrt(len(histodata)))

    # Calculate the bin edges
    bin_edges = np.linspace(np.min(histodata), np.max(histodata), num_bins+1)

    # Calculate the bin widths
    bin_widths = np.diff(bin_edges)

    # Calculate the bin heights
    bin_heights, _ = np.histogram(histodata, bins=bin_edges)

    # Normalize the bin heights so that the area of each bin is 1
    bin_heights = bin_heights / (bin_widths * len(histodata))

    # Plot the histogram
    plt.bar(bin_edges[:-1], bin_heights, width=bin_widths,alpha=0.5, edgecolor='black')
    plt.title('Equal-Area Histogram')
    plt.xlabel('Data')
    plt.ylabel('Frequency')
    plt.show()

#Creates the title for the Histogram section
histogramlist = Label(histoframe1, text = "Histograms")
histogramlist.config(font =("ariel", 18))
histogramlist.pack(pady=5)

#Creats the drop down box for the user to select the desired column
histodropdown1label = Label(histoframe1, text = "Select a column from the list:")
histodropdown1label.config(font =("ariel", 11))
histodropdown1label.pack(pady=5)
histodropdown1 = ttk.Combobox(histoframe1,state="readonly",values=['age','sex','test_time','motor_updrs','total_updrs','jitter_percent', 'jitter_abs', 'jitter_rap', 'jitter_ppq5', 'jitter_ddp', 'shimmer_percent', 'shimmer_abs', 'shimmer_apq3', 'shimmer_apq5', 'shimmer_apq11', 'shimmer_dda', 'nhr', 'hnr', 'rpde', 'dfa', 'ppe'])
histodropdown1.pack(pady=5)

#Makes a button for the user to generate the Histogram
histoselectionlistbutton= tk.Button (histoframe1, text='Generate',command=histogram1, bg='light blue')
histoselectionlistbutton.pack()
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
#histogram per subject
#Histogram function to generate a histogram based on the desired column.
def histogram1sub ():
    bananas2 = int(histosubjectent.get())
    subjectnumber2 = df[df['subject#'] == bananas2]
    histodata3 = subjectnumber2[histodropdown1sub.get()]
    
    # Calculate the number of bins
    num_bins = int(np.sqrt(len(subjectnumber2)))

    # Calculate the bin edges
    bin_edges = np.linspace(np.min(subjectnumber2), np.max(subjectnumber2), num_bins+1)

    # Calculate the bin widths
    bin_widths = np.diff(bin_edges)

    # Calculate the bin heights
    bin_heights, _ = np.histogram(subjectnumber2, bins=bin_edges)

    # Normalize the bin heights so that the area of each bin is 1
    bin_heights = bin_heights / (bin_widths * len(subjectnumber2))

    # Plot the histogram
    plt.bar(bin_edges[:-1], bin_heights, width=bin_widths,alpha=0.5, edgecolor='black')
    plt.title('Equal-Area Histogram')
    plt.xlabel('Data')
    plt.ylabel('Frequency')
    plt.show()

#Creates the title for the Histogram section
histogramlistsub = Label(histoframe2, text = "Histograms per Subject")
histogramlistsub.config(font =("ariel", 18))
histogramlistsub.pack(pady=5)

#Creats the drop down box for the user to select the desired column
histodropdown1sublabel = Label(histoframe2, text = "Select a column from the list:")
histodropdown1sublabel.config(font =("ariel", 11))
histodropdown1sublabel.pack(pady=5)
histodropdown1sub = ttk.Combobox(histoframe2,state="readonly",values=['age','sex','test_time','motor_updrs','total_updrs','jitter_percent', 'jitter_abs', 'jitter_rap', 'jitter_ppq5', 'jitter_ddp', 'shimmer_percent', 'shimmer_abs', 'shimmer_apq3', 'shimmer_apq5', 'shimmer_apq11', 'shimmer_dda', 'nhr', 'hnr', 'rpde', 'dfa', 'ppe'])
histodropdown1sub.pack(pady=5)

histosubjectentlabel = Label(histoframe2, text = "Enter Subject Number")
histosubjectentlabel.config(font =("ariel", 11))
histosubjectentlabel.pack(pady=5)
histosubjectent = tk.Entry (histoframe2) 
histosubjectent.pack(pady=5)

#Makes a button for the user to generate the Histogram
histoselectionlistbuttonsub= tk.Button (histoframe2, text='Generate',command=histogram1sub, bg='light blue')
histoselectionlistbuttonsub.pack()


###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################
#Linear Regression model

xcolumns = {'age' :'2' ,'sex':'3','test_time':'4','motor_updrs':'5','total_updrs':'6','jitter_percent':'7', 'jitter_abs':'8', 'jitter_rap':'9', 'jitter_ppq5':'10', 'jitter_ddp':'11', 'shimmer_percent':'12', 'shimmer_abs':'13', 'shimmer_apq3':'14', 'shimmer_apq5':'15', 'shimmer_apq11':'16', 'shimmer_dda':'17', 'nhr':'18', 'hnr':'19', 'rpde':'20', 'dfa':'21', 'ppe':'22'}
ycolumns = {'age' :'1' ,'sex':'2','test_time':'3','motor_updrs':'4','total_updrs':'5','jitter_percent':'6', 'jitter_abs':'7', 'jitter_rap':'8', 'jitter_ppq5':'9', 'jitter_ddp':'10', 'shimmer_percent':'11', 'shimmer_abs':'12', 'shimmer_apq3':'13', 'shimmer_apq5':'14', 'shimmer_apq11':'15', 'shimmer_dda':'16', 'nhr':'17', 'hnr':'18', 'rpde':'19', 'dfa':'20', 'ppe':'21'}
percents = {'10':0.1, '20':0.2, '30':0.3, '40':0.4, '50':0.5, '60':0.6, '70':0.7, '80':0.80, '90':0.9}


def simplelinregress():
    #Creates a list so the results can be printed in a window
    datalist = []
    #Takes the users selection for use in the linear regression calculations
    xing = int(xcolumns[xdets1.get()])
    ying = int(ycolumns[ydets1.get()])
    percenting = percents[percetage1.get()]

    # Separate explanatory variables (x) from the response variable (y)
    x = df.iloc[:,(xing-1):xing].values
    y = df.iloc[:,ying].values

    # print(x)
    # print(xing-1)
    # print(y)
    # Split dataset into 60% training and 40% test sets 
    # Note: other % split can be used.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=percenting, random_state=0)

    # Build a linear regression model
    model = LinearRegression()

    # Train (fit) the linear regression model using the training set
    model.fit(x_train, y_train)

    # Print the intercept and coefficient learned by the linear regression model
    datalist.append(("Intercept: ",model.intercept_))
    datalist.append(("\nCoefficient: ",model.coef_))

    # Use linear regression to predict the values of (y) in the test set
    # based on the values of x in the test set
    y_pred = model.predict(x_test)

    # Optional: Show the predicted values of (y) next to the actual values of (y)
    df_pred = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    datalist.append(('\n',df_pred,'\n'))

    # Compute standard performance metrics of the linear regression:

    # Mean Absolute Error
    mae = metrics.mean_absolute_error(y_test, y_pred)
    # Mean Squared Error
    mse = metrics.mean_squared_error(y_test, y_pred)
    # Root Mean Square Error
    rmse =  math.sqrt(metrics.mean_squared_error(y_test, y_pred))
    # Normalised Root Mean Square Error
    y_max = y_test.max()
    y_min = y_test.min()
    rmse_norm = rmse / (y_max - y_min)

    datalist.append(("\nMAE: ",mae))
    datalist.append(("\nMSE: ",mse))
    datalist.append(("\nRMSE: ",rmse))
    datalist.append(("\nRMSE (Normalised): ",rmse_norm))


    datalist.append("\n\n##### BASELINE MODEL #####\n")

    # Compute mean of values in (y) training set
    y_base = np.mean(y_train)

    # Replicate the mean values as many times as there are values in the test set
    y_pred_base = [y_base] * len(y_test)


    # Optional: Show the predicted values of (y) next to the actual values of (y)
    df_base_pred = pd.DataFrame({"Actual": y_test, "Predicted": y_pred_base})
    datalist.append(df_base_pred)

    # Compute standard performance metrics of the baseline model:

    # Mean Absolute Error
    mae = metrics.mean_absolute_error(y_test, y_pred_base)
    # Mean Squared Error
    mse = metrics.mean_squared_error(y_test, y_pred_base)
    # Root Mean Square Error
    rmse =  math.sqrt(metrics.mean_squared_error(y_test, y_pred_base))
    # Normalised Root Mean Square Error
    rmse_norm = rmse / (y_max - y_min)

    datalist.append("\n MAE: ")
    datalist.append((mae, '\n'))
    datalist.append("MSE: ")
    datalist.append((mse, '\n'))
    datalist.append("RMSE: ")
    datalist.append((rmse, '\n'))
    datalist.append("RMSE (Normalised): ")
    datalist.append((rmse_norm, '\n'))


    messagebox.showinfo('Linear Regression Models', datalist)
    

linregresstitle = Label()
linregresstitle = Label(linregress, text = "Simple Linear Regression\nPrediction Model")
linregresstitle.config(font =("ariel", 18))
linregresstitle.pack(pady=5)


tk.Label(linregress, text='Explanatory Variables\n(Independent Variable X-Axis)\n(Predictors)', bd=3).pack()
xdets1= tk.StringVar()
xdets2 = ttk.Combobox(linregress, values=list(xcolumns.keys()), justify="center", textvariable=xdets1)
xdets2.bind('<<ComboboxSelected>>', lambda event: xlabel.config(text=xcolumns[xdets1.get()]))
xdets2.pack()
xdets2.current(0)


xlabel = tk.Label(linregress, text="Not Selected")
xlabel.pack()


tk.Label(linregress, text='Response Variables\n(Dependent Variable Y-Axis)\n(Criterion)', bd=3).pack()
ydets1= tk.StringVar()
ydets2 = ttk.Combobox(linregress, values=list(ycolumns.keys()), justify="center", textvariable=ydets1)
ydets2.bind('<<ComboboxSelected>>', lambda event: ylabel.config(text=ycolumns[ydets1.get()]))
ydets2.pack()
ydets2.current(0)


ylabel = tk.Label(linregress, text="Not Selected")
ylabel.pack()


tk.Label(linregress, text='Test Size\nPercentage of Dataset for Testing', bd=3).pack()
percetage1= tk.StringVar()
percetage2 = ttk.Combobox(linregress, values=list(percents.keys()), justify="center", textvariable=percetage1)
percetage2.bind('<<ComboboxSelected>>', lambda event: percetagelabel.config(text=percents[percetage1.get()]))
percetage2.pack()
percetage2.current(0)


percetagelabel = tk.Label(linregress, text="Not Selected")
percetagelabel.pack()


linregressbutton= tk.Button (linregress, text='Generate',command=simplelinregress, bg='light blue')
linregressbutton.pack()

###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################

anotherdf = pd.read_csv('po2_data.csv')
column_names = ['subject#','age','sex','test_time','motor_updrs','total_updrs','jitter_percent', 'jitter_abs', 'jitter_rap', 'jitter_ppq5', 'jitter_ddp', 'shimmer_percent', 'shimmer_abs', 'shimmer_apq3', 'shimmer_apq5', 'shimmer_apq11', 'shimmer_dda', 'nhr', 'hnr', 'rpde', 'dfa', 'ppe']
anotherdf.columns= column_names

ycolumnsmulti = {'age' :'1' ,'sex':'2','test_time':'3','motor_updrs':'4','total_updrs':'5','jitter_percent':'6', 'jitter_abs':'7', 'jitter_rap':'8', 'jitter_ppq5':'9', 'jitter_ddp':'10', 'shimmer_percent':'11', 'shimmer_abs':'12', 'shimmer_apq3':'13', 'shimmer_apq5':'14', 'shimmer_apq11':'15', 'shimmer_dda':'16', 'nhr':'17', 'hnr':'18', 'rpde':'19', 'dfa':'20', 'ppe':'21'}
multipercents = {'10':0.1, '20':0.2, '30':0.3, '40':0.4, '50':0.5, '60':0.6, '70':0.7, '80':0.80, '90':0.9}

multilinregresstitled = Label(multilinregresstitle, text = "Multiple Linear Regression\nPrediction Model")
multilinregresstitled.config(font =("ariel", 18))
multilinregresstitled.pack(pady=5)

tk.Label(multilinregresstitle, text='Explanatory Variables\n(Independent Variable X-Axis)\n(Predictors)\n(Select 1 or More)', bd=3).pack()

def process(var,text):
    try:
        val = int(var.get()) 
    except ValueError:
        val = var.get()

    if val: 
        selectedcolumnlstforlinregess.append(val)
    else: 
        selectedcolumnlstforlinregess.remove(text)
        

    print(selectedcolumnlstforlinregess)

selectedcolumnlstforlinregess = []
columnsforlinregress = ['subject#','age','sex','test_time','motor_updrs','total_updrs','jitter_percent', 'jitter_abs', 'jitter_rap', 'jitter_ppq5', 'jitter_ddp', 'shimmer_percent', 'shimmer_abs', 'shimmer_apq3', 'shimmer_apq5', 'shimmer_apq11', 'shimmer_dda', 'nhr', 'hnr', 'rpde', 'dfa', 'ppe']

for idx,i in enumerate(columnsforlinregress):
    var = StringVar(value=" ")
    Checkbutton(multilinregresstitle,text=i,variable=var,command=lambda i=i,var=var: process(var,i),onvalue=i).pack()


tk.Label(multilinregresstitle, text='Response Variables\n(Dependent Variable Y-Axis)\n(Criterion)', bd=3).pack()
multiydets1= tk.StringVar()
multiydets2 = ttk.Combobox(multilinregresstitle, values=list(ycolumnsmulti.keys()), justify="center", textvariable=multiydets1)
multiydets2.bind('<<ComboboxSelected>>', lambda event: multiylabel.config(text=ycolumnsmulti[multiydets1.get()]))
multiydets2.pack(pady=5)
multiydets2.current(0)
multiylabel = tk.Label(multilinregresstitle, text="Not Selected")
multiying = int(ycolumnsmulti[multiydets1.get()])

multiuregressionlist=[]

def multilenregression():
    wtf = anotherdf[selectedcolumnlstforlinregess]
    ying = int(ycolumnsmulti[multiydets1.get()])
    multipercenting = multipercents[multipercetage1.get()]

    x = wtf
    y = df.iloc[:,ying].values

    # Split dataset into 60% training and 40% test sets 
    # Note: other % split can be used.
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=multipercenting, random_state=0)

    # Build a linear regression model
    model = LinearRegression()

    # Train (fit) the linear regression model using the training set
    model.fit(X_train, y_train)

    # Print the intercept and coefficient learned by the linear regression model
    multiuregressionlist.append(("Intercept: ", model.intercept_,'\n'))
    multiuregressionlist.append(("Coefficient: ", model.coef_,'\n'))
    print("Intercept: ", model.intercept_)
    print("Coefficient: ", model.coef_)
    # Use linear regression to predict the values of (y) in the test set
    # based on the values of x in the test set
    y_pred = model.predict(X_test)

    # Optional: Show the predicted values of (y) next to the actual values of (y)
    df_pred = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    multiuregressionlist.append(('\n',df_pred,'\n'))
    print(df_pred)
    # Compute standard performance metrics of the linear regression:

    # Mean Absolute Error
    mae = metrics.mean_absolute_error(y_test, y_pred)
    # Mean Squared Error
    mse = metrics.mean_squared_error(y_test, y_pred)
    # Root Mean Square Error
    rmse =  math.sqrt(metrics.mean_squared_error(y_test, y_pred))
    # Normalised Root Mean Square Error
    y_max = y.max()
    y_min = y.min()
    rmse_norm = rmse / (y_max - y_min)

    # R-Squared
    r_2 = metrics.r2_score(y_test, y_pred)

    multiuregressionlist.append(("\n\nMLP performance:\n"))
    multiuregressionlist.append(("MAE: ", mae,'\n'))
    multiuregressionlist.append(("MSE: ", mse,'\n'))
    multiuregressionlist.append(("RMSE: ", rmse,'\n'))
    multiuregressionlist.append(("RMSE (Normalised): ", rmse_norm,'\n'))
    multiuregressionlist.append(("R^2: ", r_2,'\n'))
    print("MLP performance:")
    print("MAE: ", mae)
    print("MSE: ", mse)
    print("RMSE: ", rmse)
    print("RMSE (Normalised): ", rmse_norm)
    print("R^2: ", r_2)


    """
    COMPARE THE PERFORMANCE OF THE LINEAR REGRESSION MODEL
    VS.
    A DUMMY MODEL (BASELINE) THAT USES MEAN AS THE BASIS OF ITS PREDICTION
    """

    # Compute mean of values in (y) training set
    y_base = np.mean(y_train)

    # Replicate the mean values as many times as there are values in the test set
    y_pred_base = [y_base] * len(y_test)


    # Optional: Show the predicted values of (y) next to the actual values of (y)
    df_base_pred = pd.DataFrame({"Actual": y_test, "Predicted": y_pred_base})
    multiuregressionlist.append(('\n',df_base_pred))

    # Compute standard performance metrics of the baseline model:

    # Mean Absolute Error
    mae = metrics.mean_absolute_error(y_test, y_pred_base)
    # Mean Squared Error
    mse = metrics.mean_squared_error(y_test, y_pred_base)
    # Root Mean Square Error
    rmse =  math.sqrt(metrics.mean_squared_error(y_test, y_pred_base))

    # Normalised Root Mean Square Error
    y_max = y.max()
    y_min = y.min()
    rmse_norm = rmse / (y_max - y_min)

    # R-Squared
    r_2 = metrics.r2_score(y_test, y_pred_base)

    multiuregressionlist.append("\n\nBaseline performance:\n")
    multiuregressionlist.append(("MAE: ", mae,'\n'))
    multiuregressionlist.append(("MSE: ", mse,'\n'))
    multiuregressionlist.append(("RMSE: ", rmse,'\n'))
    multiuregressionlist.append(("RMSE (Normalised): ", rmse_norm,'\n'))
    multiuregressionlist.append(("R^2: ", r_2,'\n'))
    print('Baseline performance:')
    print("MAE: ", mae)
    print("MSE: ", mse)
    print("RMSE: ", rmse)
    print("RMSE (Normalised): ", rmse_norm)
    print("R^2: ", r_2,)
    messagebox.showinfo('Linear Regression Models', multiuregressionlist)

multilinregressbutton= tk.Button (multilinregresstitle, text='Generate',command=multilenregression, bg='light blue')
multilinregressbutton.pack()

tk.Label(linregress, text='Test Size\nPercentage of Dataset for Testing', bd=3).pack()
multipercetage1= tk.StringVar()
multipercetage2 = ttk.Combobox(multilinregresstitle, values=list(multipercents.keys()), justify="center", textvariable=multipercetage1)
multipercetage2.bind('<<ComboboxSelected>>', lambda event: multipercetagelabel.config(text=multipercents[multipercetage1.get()]))
multipercetage2.pack()
multipercetage2.current(0)
multipercetagelabel = tk.Label(linregress, text="Not Selected")


###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################

ycolumnsmore = {'age' :'1' ,'sex':'2','test_time':'3','motor_updrs':'4','total_updrs':'5','jitter_percent':'6', 'jitter_abs':'7', 'jitter_rap':'8', 'jitter_ppq5':'9', 'jitter_ddp':'10', 'shimmer_percent':'11', 'shimmer_abs':'12', 'shimmer_apq3':'13', 'shimmer_apq5':'14', 'shimmer_apq11':'15', 'shimmer_dda':'16', 'nhr':'17', 'hnr':'18', 'rpde':'19', 'dfa':'20', 'ppe':'21'}


morelinregresstitled = Label(multilinregressmore, text = "Multiple Linear Regression With Rescale\nPrediction Model")
morelinregresstitled.config(font =("ariel", 18))
morelinregresstitled.pack(pady=5)

tk.Label(multilinregressmore, text='Explanatory Variables\n(Independent Variable X-Axis)\n(Predictors)\n(Select 1 or More)', bd=3).pack()

def moreprocess(var,text):
    try:
        val = int(var.get()) 
    except ValueError:
        val = var.get()

    if val: 
        selectedcolumnlstformorelinregess.append(val)
    else: 
        selectedcolumnlstformorelinregess.remove(text)
        

    print(selectedcolumnlstformorelinregess)

selectedcolumnlstformorelinregess = []
columnsforlinregressmore = ['subject#','age','sex','test_time','motor_updrs','total_updrs','jitter_percent', 'jitter_abs', 'jitter_rap', 'jitter_ppq5', 'jitter_ddp', 'shimmer_percent', 'shimmer_abs', 'shimmer_apq3', 'shimmer_apq5', 'shimmer_apq11', 'shimmer_dda', 'nhr', 'hnr', 'rpde', 'dfa', 'ppe']

for idx,i in enumerate(columnsforlinregressmore):
    var = StringVar(value=" ")
    Checkbutton(multilinregressmore,text=i,variable=var,command=lambda i=i,var=var: moreprocess(var,i),onvalue=i).pack()


tk.Label(multilinregressmore, text='Response Variables\n(Dependent Variable Y-Axis)\n(Criterion)', bd=3).pack()
moreydets1= tk.StringVar()
moreydets2 = ttk.Combobox(multilinregressmore, values=list(ycolumnsmore.keys()), justify="center", textvariable=moreydets1)
moreydets2.bind('<<ComboboxSelected>>', lambda event: moreylabel.config(text=ycolumnsmore[moreydets1.get()]))
moreydets2.pack(pady=5)
moreydets2.current(0)
moreylabel = tk.Label(multilinregressmore, text="Not Selected")
moreying = int(ycolumnsmore[moreydets1.get()])


moreresgress = []

def moreregression():
    # separate explanatory variables (x) from the response variable (y)
    
    
    
    x = df[selectedcolumnlstformorelinregess]
    
    morey = int(ycolumnsmore[moreydets1.get()])

    y= df.iloc[:,morey].values

    moreresgress.append(('\n',x,'\n'))
    # build and evaluate the linear regression model
    x = sm.add_constant(x) 
    model = sm.OLS(y,x).fit()
    pred = model.predict(x)
    model_details = model.summary()
    moreresgress.append(('\n',model_details,'\n'))


    """
    APPLY Z-SCORE STANDARDISATION
    """
    scaler = StandardScaler()

    # Drop the previously added constant
    x = x.drop(["const"], axis=1)

    # Apply z-score standardisation to all explanatory variables
    std_x = scaler.fit_transform(x.values)

    # Restore the column names of each explanatory variable
    std_x_df = pd.DataFrame(std_x, index=x.index, columns=x.columns)

    """
    REBUILD AND REEVALUATE LINEAR REGRESSION USING STATSMODELS
    USING STANDARDISED EXPLANATORY VARIABLES
    """

    # Build and evaluate the linear regression model
    std_x_df = sm.add_constant(std_x_df)

    moreresgress.append(('\n',std_x_df,'\n'))
    model = sm.OLS(y,std_x_df).fit()
    pred = model.predict(std_x_df)
    model_details = model.summary()
    moreresgress.append(('\n',model_details))

    messagebox.showinfo('Linear Regression Models', moreresgress)

moreregressbutton= tk.Button (multilinregressmore, text='Generate',command= moreregression, bg='light blue')
moreregressbutton.pack()

###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################

xcolumnsbysubject = {'age' :'2' ,'sex':'3','test_time':'4','motor_updrs':'5','total_updrs':'6','jitter_percent':'7', 'jitter_abs':'8', 'jitter_rap':'9', 'jitter_ppq5':'10', 'jitter_ddp':'11', 'shimmer_percent':'12', 'shimmer_abs':'13', 'shimmer_apq3':'14', 'shimmer_apq5':'15', 'shimmer_apq11':'16', 'shimmer_dda':'17', 'nhr':'18', 'hnr':'19', 'rpde':'20', 'dfa':'21', 'ppe':'22'}
ycolumnsbysubject = {'age' :'1' ,'sex':'2','test_time':'3','motor_updrs':'4','total_updrs':'5','jitter_percent':'6', 'jitter_abs':'7', 'jitter_rap':'8', 'jitter_ppq5':'9', 'jitter_ddp':'10', 'shimmer_percent':'11', 'shimmer_abs':'12', 'shimmer_apq3':'13', 'shimmer_apq5':'14', 'shimmer_apq11':'15', 'shimmer_dda':'16', 'nhr':'17', 'hnr':'18', 'rpde':'19', 'dfa':'20', 'ppe':'21'}
percentsbysubject = {'10':0.1, '20':0.2, '30':0.3, '40':0.4, '50':0.5, '60':0.6, '70':0.7, '80':0.80, '90':0.9}





def simplelinregressbysubject():
    #Creates a list so the results can be printed in a window
    datalist = []
    #Takes the users selection for use in the linear regression calculations
    bananas8 = int(subjectlinent.get())
    subjectnumber8 = df[df['subject#'] == bananas8]
    # var3 = subjectnumber1[dropdown1bysubject.get()]
    # var4 = subjectnumber1[dropdown2bysubject.get()]
  
    xingsub = int(xcolumnsbysubject[xdets1bysubject.get()])
    yingsub = int(ycolumnsbysubject[ydets1bysubject.get()])
    percenting = percentsbysubject[percetage1bysubject.get()]

    # Separate explanatory variables (x) from the response variable (y)
    x = subjectnumber8.iloc[:,(xingsub-1):xingsub].values
    y = subjectnumber8.iloc[:,yingsub].values

    # print(x)
    # print(xing-1)
    # print(y)
    # Split dataset into 60% training and 40% test sets 
    # Note: other % split can be used.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=percenting, random_state=0)

    # Build a linear regression model
    model = LinearRegression()

    # Train (fit) the linear regression model using the training set
    model.fit(x_train, y_train)

    # Print the intercept and coefficient learned by the linear regression model
    datalist.append(("Intercept: ",model.intercept_))
    datalist.append(("\nCoefficient: ",model.coef_))

    # Use linear regression to predict the values of (y) in the test set
    # based on the values of x in the test set
    y_pred = model.predict(x_test)

    # Optional: Show the predicted values of (y) next to the actual values of (y)
    df_pred = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    datalist.append(('\n',df_pred,'\n'))

    # Compute standard performance metrics of the linear regression:

    # Mean Absolute Error
    mae = metrics.mean_absolute_error(y_test, y_pred)
    # Mean Squared Error
    mse = metrics.mean_squared_error(y_test, y_pred)
    # Root Mean Square Error
    rmse =  math.sqrt(metrics.mean_squared_error(y_test, y_pred))
    # Normalised Root Mean Square Error
    y_max = y_test.max()
    y_min = y_test.min()
    rmse_norm = rmse / (y_max - y_min)

    datalist.append(("\nMAE: ",mae))
    datalist.append(("\nMSE: ",mse))
    datalist.append(("\nRMSE: ",rmse))
    datalist.append(("\nRMSE (Normalised): ",rmse_norm))


    datalist.append("\n\n##### BASELINE MODEL #####\n")

    # Compute mean of values in (y) training set
    y_base = np.mean(y_train)

    # Replicate the mean values as many times as there are values in the test set
    y_pred_base = [y_base] * len(y_test)


    # Optional: Show the predicted values of (y) next to the actual values of (y)
    df_base_pred = pd.DataFrame({"Actual": y_test, "Predicted": y_pred_base})
    datalist.append(df_base_pred)

    # Compute standard performance metrics of the baseline model:

    # Mean Absolute Error
    mae = metrics.mean_absolute_error(y_test, y_pred_base)
    # Mean Squared Error
    mse = metrics.mean_squared_error(y_test, y_pred_base)
    # Root Mean Square Error
    rmse =  math.sqrt(metrics.mean_squared_error(y_test, y_pred_base))
    # Normalised Root Mean Square Error
    rmse_norm = rmse / (y_max - y_min)

    datalist.append("\n MAE: ")
    datalist.append((mae, '\n'))
    datalist.append("MSE: ")
    datalist.append((mse, '\n'))
    datalist.append("RMSE: ")
    datalist.append((rmse, '\n'))
    datalist.append("RMSE (Normalised): ")
    datalist.append((rmse_norm, '\n'))


    messagebox.showinfo('Linear Regression Models', datalist)
    m = Message(linregressbysubject)
    txt = Text(linregressbysubject, background=m.cget("background"), relief="flat",
    borderwidth=0, font=m.cget("font"), state="disabled")
    m.destroy()


linregresstitlebysubject = Label()
linregresstitlebysubject = Label(linregressbysubject, text = "Simple Linear Regression\nPrediction Model\nBy Individual Subject")
linregresstitlebysubject.config(font =("ariel", 18))
linregresstitlebysubject.pack(pady=5)


tk.Label(linregressbysubject, text='Explanatory Variables\n(Independent Variable X-Axis)\n(Predictors)', bd=3).pack()
xdets1bysubject= tk.StringVar()
xdets2bysubject = ttk.Combobox(linregressbysubject, values=list(xcolumnsbysubject.keys()), justify="center", textvariable=xdets1bysubject)
xdets2bysubject.bind('<<ComboboxSelected>>', lambda event: xlabelbysubject.config(text=xcolumnsbysubject[xdets1bysubject.get()]))
xdets2bysubject.pack()
xdets2bysubject.current(0)


xlabelbysubject = tk.Label(linregressbysubject, text="Not Selected")
xlabelbysubject.pack()


tk.Label(linregressbysubject, text='Response Variables\n(Dependent Variable Y-Axis)\n(Criterion)', bd=3).pack()
ydets1bysubject= tk.StringVar()
ydets2bysubject = ttk.Combobox(linregressbysubject, values=list(ycolumns.keys()), justify="center", textvariable=ydets1bysubject)
ydets2bysubject.bind('<<ComboboxSelected>>', lambda event: ylabelbysubject.config(text=ycolumnsbysubject[ydets1bysubject.get()]))
ydets2bysubject.pack()
ydets2bysubject.current(0)


ylabelbysubject = tk.Label(linregressbysubject, text="Not Selected")
ylabelbysubject.pack()


tk.Label(linregressbysubject, text='Test Size\nPercentage of Dataset for Testing', bd=3).pack()
percetage1bysubject= tk.StringVar()
percetage2bysubject = ttk.Combobox(linregressbysubject, values=list(percents.keys()), justify="center", textvariable=percetage1bysubject)
percetage2bysubject.bind('<<ComboboxSelected>>', lambda event: percetagelabelbysubject.config(text=percentsbysubject[percetage1bysubject.get()]))
percetage2bysubject.pack()
percetage2bysubject.current(0)


percetagelabelbysubject = tk.Label(linregressbysubject, text="Not Selected")
percetagelabelbysubject.pack()

subjectlinentlabel = Label(linregressbysubject, text = "Enter Subject Number")
subjectlinentlabel.config(font =("ariel", 11))
subjectlinentlabel.pack(pady=5)
subjectlinent = tk.Entry (linregressbysubject) 
subjectlinent.pack(pady=5)

linregressbuttonbysubject= tk.Button (linregressbysubject, text='Generate',command=simplelinregressbysubject, bg='light blue')
linregressbuttonbysubject.pack()


###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################


#Footnote label
footnotes = Label(footerframe, text = "Thank you for using our program! \nGuy, Mark, Paulino & Nicole \nCDU\nHIT140 Dr.Yakub Sebastian \nSemester 2 2023")
footnotes.config(font =("Comic Sans MS", 20, "italic"))
footnotes.pack(pady=50)

#Runs the program on a loop so the window stays open
root.mainloop()