#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 19:25:59 2019

@author: kameokashinichi
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import shutil
from pylab import cm
from statistics import mean, stdev, median
from sklearn import linear_model
import re
from sklearn.metrics import mean_squared_error

"""
190316 test the availability of the calibrated parameter by using low-density data

variety -> SD2

calibration pattern
1. using koshihikari parameter -> "parameter_evaluation/2019-03-14_SD2_koshi_2018/Summary.OUT"
2. using observed result -> "Low_density/JPRISD01.RIA"
3. using SD1_with_HWAM parameter -> "parameter_evaluation/2019-03-14_SD2_with_HWAH_2018/Summary.OUT"
4. using SD1_without_HWAM parameter -> "parameter_evaluation/2019-03-14_SD2_without_HWAH_2018/Summary.OUT"
"""
"""
"parameter_evaluation/2019-03-14_SD2_koshi_2018/Summary.OUT"(before calibration)
"""

sd2 = pd.read_csv("Low_density/190306_sumichem_SD2_with_weatherID.csv")
sd2_2018 = sd2[sd2["year"]==2018]


record = []
with open(os.getcwd() + "/parameter_evaluation/2019-03-14_SD2_koshi_2018/Summary.OUT", 'r') as f:
    for row in f:
        record.append(row.strip())
        
summary = []
for i in range(4, len(record)):
    rec = record[i].split()
    summary.append(rec)

col = record[3].split()[1:]

summary = np.asarray(summary)
sum_df = pd.DataFrame(summary[:, 1:], index=summary[:, 0], columns=col[1:])

val = sum_df[["HWAM", "ADAT", "MDAT", "PDAT"]].values
val = np.concatenate((val, sd2_2018["prefecture"].values.reshape(-1,1)), axis=1)
sim_df = pd.DataFrame(val, index=summary[:, 0].astype(np.int32), columns=["HWAM", "ADAT", "MDAT", "PDAT", "PREF"])

"""
"Low_density/JPRI1802.RIA" (measured value)
"""
record = []
with open(os.getcwd() + "/Low_density/JPRI1802.RIA", 'r') as f:
    for row in f:
        record.append(row.strip())
        
summary = []
for i in range(3, len(record)):
    rec = record[i].split()
    summary.append(rec)
    
col = record[2].split()[1:]

summary = np.asarray(summary)
real_df = pd.DataFrame(summary[:, 1:], index=summary[:, 0].astype(np.int32), columns=col)

"""
"parameter_evaluation/2019-03-14_SD2_with_HWAH_2018/Summary.OUT" (calibrate including HWAM)
"""
record = []
with open(os.getcwd() + "/parameter_evaluation/2019-03-14_SD2_with_HWAH_2018/Summary.OUT", 'r') as f:
    for row in f:
        record.append(row.strip())
        
summary = []
for i in range(4, len(record)):
    rec = record[i].split()
    summary.append(rec)

col = record[3].split()[1:]

summary = np.asarray(summary)
sum_df = pd.DataFrame(summary[:, 1:], index=summary[:, 0], columns=col[1:])

val = sum_df[["HWAM", "ADAT", "MDAT", "PDAT"]].values
val = np.concatenate((val, sd2_2018["prefecture"].values.reshape(-1,1)), axis=1)
calh_df = pd.DataFrame(val, index=summary[:, 0].astype(np.int32), columns=["HWAM", "ADAT", "MDAT", "PDAT", "PREF"])

"""
"parameter_evaluation/2019-03-14_SD2_without_HWAH_2018/Summary.OUT" (calibrate without HWAM)
"""
record = []
with open(os.getcwd() + "/parameter_evaluation/2019-03-14_SD2_without_HWAH_2018/Summary.OUT", 'r') as f:
    for row in f:
        record.append(row.strip())
        
summary = []
for i in range(4, len(record)):
    rec = record[i].split()
    summary.append(rec)

col = record[3].split()[1:]

summary = np.asarray(summary)
sum_df = pd.DataFrame(summary[:, 1:], index=summary[:, 0], columns=col[1:])

val = sum_df[["HWAM", "ADAT", "MDAT", "PDAT"]].values
val = np.concatenate((val, sd2_2018["prefecture"].values.reshape(-1,1)), axis=1)
cal_df = pd.DataFrame(val, index=summary[:, 0].astype(np.int32), columns=["HWAM", "ADAT", "MDAT", "PDAT", "PREF"])

"""
generate a dataframe of RMSE
"""
col = {"SIMULATION": sim_df, "CAL_NO_HWAM": cal_df, "CAL_WITH_HWAH": calh_df}
ind = ["ADAT", "MDAT", "HWAM"]
rmse_lis = []
df_lis = []
for i in range(len(ind)):
    l = []
    for key, val in col.items():        
        if ind[i] != "HWAM":
            print(ind[i])
            print([key])
            sim = pd.DataFrame(val[ind[i]].values, index=val.index, columns=[key])
            real = pd.DataFrame("2018" + real_df[ind[i]].values, index=real_df.index, columns=["OBSERVED"])
            mdat_df = pd.concat([sim, real], axis=1)
            #m_val = np.concatenate((sim.reshape(-1,1), real.reshape(-1,1)),axis=1)
            #mdat_df = pd.DataFrame(m_val, index=real_df.index, columns=[key, "OBSERVED"])
            
            mdat_df = mdat_df.replace({"^2017\d{3}$": pd.np.nan, "-99": pd.np.nan}, regex=True) #drop the error
            mdat_df = mdat_df.dropna()
            
            mdat = pd.DataFrame(mdat_df.values.astype(np.int32), index=mdat_df.index, columns=mdat_df.columns)
            
            pre = mdat[key].values - sim_df.loc[mdat.index, "PDAT"].values.astype(np.int) 
            obs = mdat["OBSERVED"].values - sim_df.loc[mdat.index, "PDAT"].values.astype(np.int)
            rmse = np.sqrt(mean_squared_error(pre, obs))
            
            l.append(rmse)
            df_lis.append(mdat)
            
        else:
            """
            sim = pd.DataFrame(val[ind[i]].values, index=val.index, columns=[key])
            real = pd.DataFrame(real_df[ind[i]].values, index=real_df.index, columns=["OBSERVED"])
            mdat_df = pd.concat([sim, real], axis=1)
            #m_val = np.concatenate((sim.reshape(-1,1), real.reshape(-1,1)),axis=1)
            #mdat_df = pd.DataFrame(m_val, index=real_df.index, columns=[key, "OBSERVED"])
            
            mdat_df = mdat_df.replace({"^2017\d{3}$": pd.np.nan, "-99": pd.np.nan}, regex=True) #drop the error
            mdat_df = mdat_df.dropna()
            
            mdat = pd.DataFrame(mdat_df.values.astype(np.int32), index=mdat_df.index, columns=mdat_df.columns)
            
            pre = mdat[key].values - sim_df.loc[mdat.index, "PDAT"].values.astype(np.int) 
            obs = mdat["OBSERVED"].values - sim_df.loc[mdat.index, "PDAT"].values.astype(np.int)
            rmse = np.sqrt(mean_squared_error(pre, obs))
            """
            pass
        
    rmse_lis.append(l)

        
rmse_df = pd.DataFrame(rmse_lis, index=ind, columns=col)

rmse_df.to_csv("190316_evaluate_calibration_via_RMSE_SD02_2018.csv")


"""
compare these two data(RMSE etc.)
"""
#1. ADAT
sim = sim_df["ADAT"].values
real = "2018" + real_df["ADAT"].values
cal = cal_df["ADAT"].values
calh = calh_df["ADAT"].values
pref = sim_df["PREF"].values

a_val = np.concatenate((sim.reshape(-1,1), cal.reshape(-1,1), calh.reshape(-1,1), pref.reshape(-1,1)),axis=1)
adat_df = pd.DataFrame(a_val, index=sim_df.index, columns=["SIMULATION", "CAL_NO_HWAM", "CAL_WITH_HWAH", "PREF"])
r_df = pd.DataFrame(real, index=real_df.index, columns=["OBSERVED"])
adat_df = pd.concat([r_df, adat_df], axis=1)

adat_df = adat_df.replace({"^2017\d{3}$": pd.np.nan, "-99": pd.np.nan}, regex=True) #drop the error
adat_df = adat_df.dropna()

value = np.concatenate((adat_df.values[:, :4].astype(np.int32), adat_df.values[:, 4].reshape(-1,1)), axis=1)
adat = pd.DataFrame(value, index=adat_df.index, columns=adat_df.columns)

a_rmse = np.sqrt(mean_squared_error(adat.iloc[:, 0].values, adat.iloc[:, 1].values))

#2. MDAT
sim = sim_df["MDAT"].values
real = "2018" + real_df["MDAT"].values
cal = cal_df["MDAT"].values
calh = calh_df["MDAT"].values
pref = sim_df["PREF"].values

m_val = np.concatenate((sim.reshape(-1,1), cal.reshape(-1,1), calh.reshape(-1,1), pref.reshape(-1,1)),axis=1)
mdat_df = pd.DataFrame(m_val, index=sim_df.index, columns=["SIMULATION", "CAL_NO_HWAM", "CAL_WITH_HWAH", "PREF"])
r_df = pd.DataFrame(real, index=real_df.index, columns=["OBSERVED"])
mdat_df = pd.concat([r_df, mdat_df], axis=1)

mdat_df = mdat_df.replace({"^2017\d{3}$": pd.np.nan, "-99": pd.np.nan}, regex=True) #drop the error
mdat_df = mdat_df.dropna()

value = np.concatenate((mdat_df.values[:, :4].astype(np.int32), mdat_df.values[:, 4].reshape(-1,1)), axis=1)
mdat = pd.DataFrame(value, index=mdat_df.index, columns=mdat_df.columns)

m_rmse = np.sqrt(mean_squared_error(mdat.iloc[:, 0].values, mdat.iloc[:, 1].values))

#3. HWAM
sim = sim_df["HWAM"].values
real = real_df["HWAM"].values
cal = cal_df["HWAM"].values
calh = calh_df["HWAM"].values
pref = sim_df["PREF"].values

h_val = np.concatenate((sim.reshape(-1,1), cal.reshape(-1,1), calh.reshape(-1,1), pref.reshape(-1,1)),axis=1)
hwah_df = pd.DataFrame(h_val, index=sim_df.index, columns=["SIMULATION", "CAL_NO_HWAM", "CAL_WITH_HWAH", "PREF"])
r_df = pd.DataFrame(real, index=real_df.index, columns=["OBSERVED"])
hwah_df = pd.concat([r_df, hwah_df], axis=1)

hwah_df = hwah_df.replace({"^2017\d{3}$": pd.np.nan, "-99": pd.np.nan}, regex=True) #drop the error
hwah_df = hwah_df.dropna()

value = np.concatenate((hwah_df.values[:, :4].astype(np.int32), hwah_df.values[:, 4].reshape(-1,1)), axis=1)
hwah = pd.DataFrame(value, index=hwah_df.index, columns=hwah_df.columns)

h_rmse = np.sqrt(mean_squared_error(hwah.iloc[:, 0].values, hwah.iloc[:, 1].values))


"""
190316 visualize the impact of calibration
comparing calh vs real vs sim

visualize the several point of data by bar plot
1. generate dataframe for each prefecture
2. visualize that by bar plot
"""
#generate dataframe sorted by prefecture -ADAT-
arr = []
for i in np.unique(adat["PREF"]):
    df = adat[adat["PREF"]==i]
    lis = []
    for j in range(4):
        ave = int(mean(df.iloc[:, j].values))
        lis.append(ave)
    arr.append(lis)
    
arr = np.asarray(arr)
arr = np.concatenate((arr, np.unique(adat["PREF"]).reshape(-1,1)), axis=1)

adat3 = pd.DataFrame(arr, index=np.arange(1, arr.shape[0]+1), columns=adat.columns)

#visualize ADAT
adat2 = adat3.iloc[:, [0,1,3]]

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1,1,1)
name = ["observed", "Koshihikari", "SD2"]

n = 0
for i in range(len(adat2.columns)):
    ax.bar(np.arange(len(adat2.index))+n, adat2.iloc[:, i], color=cm.hsv(i/3),
           label=name[i], width=0.8/3)
    n = n + 0.8/3
    
plt.legend(loc="best",fontsize=14)
plt.xticks(np.arange(len(adat2.index)+0.55), adat3["PREF"].values, rotation=45, fontsize=14)
plt.title("Comparison of anthesis date among observed, koshi_param and SD2_param", fontsize=18)
plt.xlabel("field name", fontsize=15)
plt.ylabel("Anthesis date", fontsize=15)
plt.ylim([min(adat2.iloc[:, 1]-10), max(adat2.iloc[:, 0]+10)])
ax.set_yticklabels(doylist2Date(ax.get_yticks() -2018000))
plt.setp(ax.get_yticklabels(), fontsize=14, rotation=45, visible=True)
plt.savefig("190313_rice_gencalc_png/190316_ADAT_validation_SD2.png", bbox_inches="tight")

plt.show()


#generate dataframe sorted by prefecture -MDAT-
arr = []
for i in np.unique(mdat["PREF"]):
    df = mdat[mdat["PREF"]==i]
    lis = []
    for j in range(4):
        ave = int(mean(df.iloc[:, j].values))
        lis.append(ave)
    arr.append(lis)
    
arr = np.asarray(arr)
arr = np.concatenate((arr, np.unique(mdat["PREF"]).reshape(-1,1)), axis=1)

mdat3 = pd.DataFrame(arr, index=np.arange(1, arr.shape[0]+1), columns=mdat.columns)

#visualize MDAT
mdat2 = mdat3.iloc[:, [0,1,3]]

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1,1,1)
name = ["observed", "Koshihikari", "SD2"]

n = 0
for i in range(len(mdat2.columns)):
    ax.bar(np.arange(len(mdat2.index))+n, mdat2.iloc[:, i], color=cm.hsv(i/3),
           label=name[i], width=0.8/3)
    n = n + 0.8/3
    
plt.legend(loc="best",fontsize=14)
plt.xticks(np.arange(len(mdat2.index)+0.55), mdat3["PREF"].values, rotation=30, fontsize=14)
plt.title("Comparison of maturity date among observed, koshi_param and SD2_param", fontsize=18)
plt.xlabel("field name", fontsize=15)
plt.ylabel("Maturity date", fontsize=15)
plt.ylim([min(mdat2.iloc[:, 1]-10), max(mdat2.iloc[:, 0]+10)])
ax.set_yticklabels(doylist2Date(ax.get_yticks() -2018000))
plt.setp(ax.get_yticklabels(), fontsize=14, rotation=45, visible=True)
plt.savefig("190313_rice_gencalc_png/190316_MDAT_validation_SD2.png", bbox_inches="tight")

plt.show()



















