#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 11:20:45 2019

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
190318 test the availability of the calibrated parameter by using low-density data

variety -> SD1

calibration pattern
1. using koshihikari parameter -> "parameter_evaluation/2019-03-18_SD1_koshi_2017/Summary.OUT"
2. using observed result -> "Low_density/JPRI1701.RIA"
3. using SD1_with_HWAM parameter -> "parameter_evaluation/2019-03-18_SD1_HWAH_2017/Summary.OUT"

"""

sd1 = pd.read_csv("Low_density/190306_sumichem_SD1_with_weatherID.csv")
sd1_2018 = sd1[sd1["year"]==2018]
sd1_2017 = sd1[sd1["year"]==2017]

"""
"parameter_evaluation/2019-03-18_SD1_koshi_2017/Summary.OUT"(before calibration)
"""
record = []
with open(os.getcwd() + "/parameter_evaluation/2019-03-18_SD1_koshi_2017/Summary.OUT", 'r') as f:
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
val = np.concatenate((val, sd1_2017["prefecture"].values.reshape(-1,1)), axis=1)
sim_df = pd.DataFrame(val, index=summary[:, 0].astype(np.int32), columns=["HWAM", "ADAT", "MDAT", "PDAT", "PREF"])

"""
"Low_density/JPRI1701.RIA" (measured value)
"""
#genCalibrationfileFromDf(sd1_2017, "Low_density/JPRI1701.RIA", crop="rice")

record = []
with open(os.getcwd() + "/Low_density/JPRI1701.RIA", 'r') as f:
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
"parameter_evaluation/2019-03-18_SD1_HWAH_2017/Summary.OUT" (calibrate including HWAM)
"""
record = []
with open(os.getcwd() + "/parameter_evaluation/2019-03-18_SD1_HWAH_2017/Summary.OUT", 'r') as f:
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
val = np.concatenate((val, sd1_2017["prefecture"].values.reshape(-1,1)), axis=1)
calh_df = pd.DataFrame(val, index=summary[:, 0].astype(np.int32), columns=["HWAM", "ADAT", "MDAT", "PDAT", "PREF"])


"""
190318 Compare the RMSE of koshihikari(sim) and SD1(calh) in terms of HWAH

1. generate dataframe which contains HWAH of sim, real and calh, in addition to prefecture
2. visualize it
"""
#3. HWAM
sim = sim_df["HWAM"].values
real = real_df["HWAM"].values
#cal = cal_df["HWAM"].values
calh = calh_df["HWAM"].values
pref = sim_df["PREF"].values

h_val = np.concatenate((sim.reshape(-1,1), calh.reshape(-1,1), pref.reshape(-1,1)),axis=1)
hwah_df = pd.DataFrame(h_val, index=sim_df.index, columns=["SIMULATION", "CAL_WITH_HWAH", "PREF"])
r_df = pd.DataFrame(real, index=real_df.index, columns=["OBSERVED"])
hwah_df = pd.concat([r_df, hwah_df], axis=1)

hwah_df = hwah_df.replace({"^2018\d{3}$": pd.np.nan, "-99": pd.np.nan}, regex=True) #drop the error
hwah_df = hwah_df.dropna()

value = np.concatenate((hwah_df.values[:, :3].astype(np.int32), hwah_df.values[:, 3].reshape(-1,1)), axis=1)
hwah = pd.DataFrame(value, index=hwah_df.index, columns=hwah_df.columns)

h_rmse = np.sqrt(mean_squared_error(hwah.iloc[:, 0].values, hwah.iloc[:, 1].values))


#generate dataframe sorted by prefecture -HWAH-
arr = []
for i in np.unique(hwah["PREF"]):
    df = hwah[hwah["PREF"]==i]
    lis = []
    for j in range(3):
        ave = int(mean(df.iloc[:, j].values))
        lis.append(ave)
    arr.append(lis)
    
arr = np.asarray(arr)
arr = np.concatenate((arr, np.unique(hwah["PREF"]).reshape(-1,1)), axis=1)

hwah3 = pd.DataFrame(arr, index=np.arange(1, arr.shape[0]+1), columns=hwah.columns)

#visualize hwah
hwah2 = hwah3.iloc[:, :3]

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1,1,1)
name = ["observed", "Koshihikari", "SD1"]

n = 0
for i in range(len(hwah2.columns)):
    ax.bar(np.arange(len(hwah2.index))+n, hwah2.iloc[:, i], color=cm.hsv(i/3),
           label=name[i], width=0.8/3)
    n = n + 0.8/3
    
plt.legend(loc="best",fontsize=14)
plt.xticks(np.arange(len(hwah2.index)+0.55), hwah3["PREF"].values, rotation=30, fontsize=14)
plt.title("Comparison of Yield among observed, koshi_param and SD1_param", fontsize=18)
plt.xlabel("field name", fontsize=15)
plt.ylabel("Yield(kg/ha)", fontsize=15)
plt.ylim([min(hwah2.iloc[:, 1])-100, max(hwah2.iloc[:, 0])+100])
#ax.set_yticklabels(doylist2Date(ax.get_yticks() -2018000))
plt.setp(ax.get_yticklabels(), fontsize=14, rotation=45, visible=True)
plt.savefig("190313_rice_gencalc_png/190318_HWAH_validation_SD1_2017.png", bbox_inches="tight")

plt.show()


"""
generate a dataframe of RMSE
"""
col = {"SIMULATION": sim_df, "CAL_WITH_HWAH": calh_df}
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
            real = pd.DataFrame("2017" + real_df[ind[i]].values, index=real_df.index, columns=["OBSERVED"])
            mdat_df = pd.concat([sim, real], axis=1)
            #m_val = np.concatenate((sim.reshape(-1,1), real.reshape(-1,1)),axis=1)
            #mdat_df = pd.DataFrame(m_val, index=real_df.index, columns=[key, "OBSERVED"])
            
            mdat_df = mdat_df.replace({"^2018\d{3}$": pd.np.nan, "-99": pd.np.nan}, regex=True) #drop the error
            mdat_df = mdat_df.dropna()
            
            mdat = pd.DataFrame(mdat_df.values.astype(np.int32), index=mdat_df.index, columns=mdat_df.columns)
            
            pre = mdat[key].values - sim_df.loc[mdat.index, "PDAT"].values.astype(np.int) 
            obs = mdat["OBSERVED"].values - sim_df.loc[mdat.index, "PDAT"].values.astype(np.int)
            rmse = np.sqrt(mean_squared_error(pre, obs))
            
        else:
            
            sim = pd.DataFrame(val[ind[i]].values, index=val.index, columns=[key])
            real = pd.DataFrame(real_df[ind[i]].values, index=real_df.index, columns=["OBSERVED"])
            mdat_df = pd.concat([sim, real], axis=1)
            #m_val = np.concatenate((sim.reshape(-1,1), real.reshape(-1,1)),axis=1)
            #mdat_df = pd.DataFrame(m_val, index=real_df.index, columns=[key, "OBSERVED"])
            
            mdat_df = mdat_df.replace({"-99": pd.np.nan}, regex=True) #drop the error
            mdat_df = mdat_df.dropna()
            
            mdat = pd.DataFrame(mdat_df.values.astype(np.int32), index=mdat_df.index, columns=mdat_df.columns)
            
            pre = mdat[key].values
            #print(pre)
            obs = mdat["OBSERVED"].values
            #print(obs)
            rmse = np.sqrt(mean_squared_error(pre, obs))
            
        l.append(rmse)
        df_lis.append(mdat)
        
    rmse_lis.append(l)

        
rmse_df = pd.DataFrame(rmse_lis, index=ind, columns=col)

rmse_df.to_csv("190318_evaluate_calibration_via_RMSE_SD01_2017.csv")



"""
190318 test the availability of the calibrated parameter by using low-density data

variety -> SD2

calibration pattern
1. using koshihikari parameter -> "parameter_evaluation/2019-03-18_SD2_koshi_2017/Summary.OUT"
2. using observed result -> "Low_density/JPRI1702.RIA"
3. using SD1_with_HWAM parameter -> "parameter_evaluation/2019-03-18_SD2_HWAH_2017/Summary.OUT"

"""

sd2 = pd.read_csv("Low_density/190306_sumichem_SD2_with_weatherID.csv")
sd2_2018 = sd2[sd2["year"]==2018]
sd2_2017 = sd2[sd2["year"]==2017]
"""
"parameter_evaluation/2019-03-18_SD2_koshi_2017/Summary.OUT"(before calibration)
"""
record = []
with open(os.getcwd() + "/parameter_evaluation/2019-03-18_SD2_koshi_2017/Summary.OUT", 'r') as f:
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
val = np.concatenate((val, sd2_2017["prefecture"].values.reshape(-1,1)), axis=1)
sim_df = pd.DataFrame(val, index=summary[:, 0].astype(np.int32), columns=["HWAM", "ADAT", "MDAT", "PDAT", "PREF"])

"""
"Low_density/JPRI1702.RIA" (measured value)
"""
#genCalibrationfileFromDf(sd2_2017, "Low_density/JPRI1702.RIA", crop="rice")

record = []
with open(os.getcwd() + "/Low_density/JPRI1702.RIA", 'r') as f:
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
"parameter_evaluation/2019-03-18_SD2_HWAH_2017/Summary.OUT" (calibrate including HWAM)
"""
record = []
with open(os.getcwd() + "/parameter_evaluation/2019-03-18_SD2_HWAH_2017/Summary.OUT", 'r') as f:
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
val = np.concatenate((val, sd2_2017["prefecture"].values.reshape(-1,1)), axis=1)
calh_df = pd.DataFrame(val, index=summary[:, 0].astype(np.int32), columns=["HWAM", "ADAT", "MDAT", "PDAT", "PREF"])


"""
190318 Compare the RMSE of koshihikari(sim) and SD2(calh) in terms of HWAH

1. generate dataframe which contains HWAH of sim, real and calh, in addition to prefecture
2. visualize it
"""
#3. HWAM
sim = sim_df["HWAM"].values
real = real_df["HWAM"].values
#cal = cal_df["HWAM"].values
calh = calh_df["HWAM"].values
pref = sim_df["PREF"].values

h_val = np.concatenate((sim.reshape(-1,1), calh.reshape(-1,1), pref.reshape(-1,1)),axis=1)
hwah_df = pd.DataFrame(h_val, index=sim_df.index, columns=["SIMULATION", "CAL_WITH_HWAH", "PREF"])
r_df = pd.DataFrame(real, index=real_df.index, columns=["OBSERVED"])
hwah_df = pd.concat([r_df, hwah_df], axis=1)

hwah_df = hwah_df.replace({"^2018\d{3}$": pd.np.nan, "-99": pd.np.nan}, regex=True) #drop the error
hwah_df = hwah_df.dropna()

value = np.concatenate((hwah_df.values[:, :3].astype(np.int32), hwah_df.values[:, 3].reshape(-1,1)), axis=1)
hwah = pd.DataFrame(value, index=hwah_df.index, columns=hwah_df.columns)

h_rmse = np.sqrt(mean_squared_error(hwah.iloc[:, 0].values, hwah.iloc[:, 1].values))


#generate dataframe sorted by prefecture -HWAH-
arr = []
for i in np.unique(hwah["PREF"]):
    df = hwah[hwah["PREF"]==i]
    lis = []
    for j in range(3):
        ave = int(mean(df.iloc[:, j].values))
        lis.append(ave)
    arr.append(lis)
    
arr = np.asarray(arr)
arr = np.concatenate((arr, np.unique(hwah["PREF"]).reshape(-1,1)), axis=1)

hwah3 = pd.DataFrame(arr, index=np.arange(1, arr.shape[0]+1), columns=hwah.columns)

#visualize hwah
hwah2 = hwah3.iloc[:, :3]

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1,1,1)
name = ["observed", "Koshihikari", "SD2"]

n = 0
for i in range(len(hwah2.columns)):
    ax.bar(np.arange(len(hwah2.index))+n, hwah2.iloc[:, i], color=cm.hsv(i/3),
           label=name[i], width=0.8/3)
    n = n + 0.8/3
    
plt.legend(loc="best",fontsize=14)
plt.xticks(np.arange(len(hwah2.index)+0.55), hwah3["PREF"].values, rotation=30, fontsize=14)
plt.title("Comparison of Yield among observed, koshi_param and SD2_param", fontsize=18)
plt.xlabel("field name", fontsize=15)
plt.ylabel("Yield(kg/ha)", fontsize=15)
plt.ylim([min(hwah2.iloc[:, 1]-100), max(hwah2.iloc[:, 0]+100)])
#ax.set_yticklabels(doylist2Date(ax.get_yticks() -2018000))
plt.setp(ax.get_yticklabels(), fontsize=14, rotation=45, visible=True)
plt.savefig("190313_rice_gencalc_png/190318_HWAH_validation_SD2.png", bbox_inches="tight")

plt.show()


"""
generate a dataframe of RMSE
"""
col = {"SIMULATION": sim_df, "CAL_WITH_HWAH": calh_df}
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
            real = pd.DataFrame("2017" + real_df[ind[i]].values, index=real_df.index, columns=["OBSERVED"])
            mdat_df = pd.concat([sim, real], axis=1)
            #m_val = np.concatenate((sim.reshape(-1,1), real.reshape(-1,1)),axis=1)
            #mdat_df = pd.DataFrame(m_val, index=real_df.index, columns=[key, "OBSERVED"])
            
            mdat_df = mdat_df.replace({"^2018\d{3}$": pd.np.nan, "-99": pd.np.nan}, regex=True) #drop the error
            mdat_df = mdat_df.dropna()
            
            mdat = pd.DataFrame(mdat_df.values.astype(np.int32), index=mdat_df.index, columns=mdat_df.columns)
            
            pre = mdat[key].values - sim_df.loc[mdat.index, "PDAT"].values.astype(np.int) 
            obs = mdat["OBSERVED"].values - sim_df.loc[mdat.index, "PDAT"].values.astype(np.int)
            rmse = np.sqrt(mean_squared_error(pre, obs))
            
        else:
            
            sim = pd.DataFrame(val[ind[i]].values, index=val.index, columns=[key])
            real = pd.DataFrame(real_df[ind[i]].values, index=real_df.index, columns=["OBSERVED"])
            mdat_df = pd.concat([sim, real], axis=1)
            #m_val = np.concatenate((sim.reshape(-1,1), real.reshape(-1,1)),axis=1)
            #mdat_df = pd.DataFrame(m_val, index=real_df.index, columns=[key, "OBSERVED"])
            
            mdat_df = mdat_df.replace({"-99": pd.np.nan}, regex=True) #drop the error
            mdat_df = mdat_df.dropna()
            
            mdat = pd.DataFrame(mdat_df.values.astype(np.int32), index=mdat_df.index, columns=mdat_df.columns)
            
            pre = mdat[key].values
            #print(pre)
            obs = mdat["OBSERVED"].values
            #print(obs)
            rmse = np.sqrt(mean_squared_error(pre, obs))
            
        l.append(rmse)
        df_lis.append(mdat)
        
    rmse_lis.append(l)

        
rmse_df = pd.DataFrame(rmse_lis, index=ind, columns=col)

rmse_df.to_csv("190318_evaluate_calibration_via_RMSE_SD02_2017.csv")




















