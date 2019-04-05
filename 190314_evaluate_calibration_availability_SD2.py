#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 09:15:52 2019

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
190314 evaluate the availability of the parameter calibration

1. YIELD(HWAM)
2. ANTHESIS DATE(ADAT)
3. MATURITY DATE(MDAT)

variety -> SD2
used parameter(990005: koshihikari)
1. before calibration
  ECO#    P1   P2R    P5   P2O    G1    G2    G3    G4 PHINT
IB0001 252.1 44.39 555.7 10.04 45.59 .0250  1.00  1.00  83.0
2. after calibration (990012: SD01_high_density with HWAH)
  ECO#    P1   P2R    P5   P2O    G1    G2    G3    G4 PHINT
IB0001 251.9 45.57 540.1 8.654 57.78 .0250  1.00  1.00  83.0
3. after calibration (990012: SD01_high_density without HWAH)
  ECO#    P1   P2R    P5   P2O    G1    G2    G3    G4 PHINT
IB0001 251.9 45.57 540.1 8.654 45.59 .0250  1.00  1.00  83.0

list of parameter name
sim_df -> dataframe of simulation result before calibration
real_df -> dataframe of measured value
calh_df -> dataframe of simulation with calibrated parameter(including HWAM)
cal_df -> dataframe of simulation with calibrated parameter(without HWAM)
"""

"""
/High_density/SD2_high_density_result/2019-03-08T10-30-48-451Zc0e56c6dff8bdbe4/Summary.OUT(before calibration)
"""

record = []
with open(os.getcwd() + '/High_density/SD1_high_density_result/2019-03-08T10-28-05-929Z99cc018dcf2ec5a3/summary.OUT', 'r') as f:
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
sim_df = pd.DataFrame(val, index=summary[:, 0].astype(np.int32), columns=["HWAM", "ADAT", "MDAT", "PDAT"])

"""
./High_density/SD2_high_density_result/JPRI0001.RIA (measured value)
"""
record = []
with open(os.getcwd() + '/High_density/SD2_high_density_result/JPRI0001.RIA', 'r') as f:
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
2019-03-13_SD02_high_density_with_HWAH/Summary.OUT (calibrate including HWAM)
"""
record = []
with open(os.getcwd() + '/2019-03-13_SD02_high_density_with_HWAH/summary.OUT', 'r') as f:
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
calh_df = pd.DataFrame(val, index=summary[:, 0].astype(np.int32), columns=["HWAM", "ADAT", "MDAT", "PDAT"])

"""
2019-03-13_SD02_high_density_without_HWAH/Summary.OUT (calibrate without HWAM)
"""
record = []
with open(os.getcwd() + '/2019-03-13_SD02_high_density_without_HWAH/summary.OUT', 'r') as f:
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
cal_df = pd.DataFrame(val, index=summary[:, 0].astype(np.int32), columns=["HWAM", "ADAT", "MDAT", "PDAT"])


"""
generate a dataframe of RMSE
"""
col = {"SIMULATION": sim_df, "CAL_NO_HWAM": cal_df, "CAL_WITH_HWAH": calh_df}
ind = ["ADAT", "MDAT", "HWAM"]
rmse_lis = []
for i in range(len(ind)):
    l = []
    for key, val in col.items():        
        if ind[i] != "HWAM":
            print(ind[i])
            sim = val[ind[i]].values
            real = "20" + real_df[ind[i]].values
            m_val = np.concatenate((sim.reshape(-1,1), real.reshape(-1,1)),axis=1)
            mdat_df = pd.DataFrame(m_val, index=real_df.index, columns=[key, "OBSERVED"])
            
            mdat_df = mdat_df.replace({"^2017\d{3}$": pd.np.nan, "-99": pd.np.nan}, regex=True) #drop the error
            mdat_df = mdat_df.dropna()
            
            mdat = pd.DataFrame(mdat_df.values.astype(np.int32), index=mdat_df.index, columns=mdat_df.columns)
            
            pre = mdat[key].values - sim_df.loc[mdat.index, "PDAT"].values.astype(np.int) 
            obs = mdat["OBSERVED"].values - sim_df.loc[mdat.index, "PDAT"].values.astype(np.int)
            rmse = np.sqrt(mean_squared_error(pre, obs))
            
        else:
            sim = val[ind[i]].values
            real = real_df[ind[i]].values
            m_val = np.concatenate((sim.reshape(-1,1), real.reshape(-1,1)),axis=1)
            mdat_df = pd.DataFrame(m_val, index=real_df.index, columns=[key, "OBSERVED"])
            
            mdat_df = mdat_df.replace({"^2017\d{3}$": pd.np.nan, "-99": pd.np.nan}, regex=True) #drop the error
            mdat_df = mdat_df.dropna()
            
            mdat = pd.DataFrame(mdat_df.values.astype(np.int32), index=mdat_df.index, columns=mdat_df.columns)
            
            pre = mdat[key].values - sim_df.loc[mdat.index, "PDAT"].values.astype(np.int) 
            obs = mdat["OBSERVED"].values - sim_df.loc[mdat.index, "PDAT"].values.astype(np.int)
            rmse = np.sqrt(mean_squared_error(pre, obs))
        
        l.append(rmse)
        
    rmse_lis.append(l)
        
rmse_df = pd.DataFrame(rmse_lis, index=ind, columns=col)

rmse_df.to_csv("190314_evaluate_calibration_via_RMSE_SD02.csv")


"""
compare these two data(RMSE etc.)
"""
#1. ADAT
sim = sim_df["ADAT"].values
real = "20" + real_df["ADAT"].values
cal = cal_df["ADAT"].values
calh = calh_df["ADAT"].values

a_val = np.concatenate((sim.reshape(-1,1), real.reshape(-1,1), cal.reshape(-1,1), calh.reshape(-1,1)),axis=1)
adat_df = pd.DataFrame(a_val, index=real_df.index, columns=["SIMULATION", "OBSERVED", "CAL_NO_HWAM", "CAL_WITH_HWAH"])

adat_df = adat_df.replace({"^2017\d{3}$": pd.np.nan, "-99": pd.np.nan}, regex=True) #drop the error
adat_df = adat_df.dropna()

adat = pd.DataFrame(adat_df.values.astype(np.int32), index=adat_df.index, columns=adat_df.columns)

a_rmse = np.sqrt(mean_squared_error(adat.iloc[:, 0].values, adat.iloc[:, 1].values))

#2. MDAT
sim = sim_df["MDAT"].values
real = "20" + real_df["MDAT"].values
cal = cal_df["MDAT"].values
calh = calh_df["MDAT"].values

m_val = np.concatenate((sim.reshape(-1,1), real.reshape(-1,1), cal.reshape(-1,1), calh.reshape(-1,1)),axis=1)
mdat_df = pd.DataFrame(m_val, index=real_df.index, columns=["SIMULATION", "OBSERVED", "CAL_NO_HWAM", "CAL_WITH_HWAH"])

mdat_df = mdat_df.replace({"^2017\d{3}$": pd.np.nan, "-99": pd.np.nan}, regex=True) #drop the error
mdat_df = mdat_df.dropna()

mdat = pd.DataFrame(mdat_df.values.astype(np.int32), index=mdat_df.index, columns=mdat_df.columns)

m_rmse = np.sqrt(mean_squared_error(mdat.iloc[:, 0].values, mdat.iloc[:, 1].values))

#3. HWAM
sim = sim_df["HWAM"].values
real = real_df["HWAM"].values
cal = cal_df["HWAM"].values
calh = calh_df["HWAM"].values

h_val = np.concatenate((sim.reshape(-1,1), real.reshape(-1,1), cal.reshape(-1,1), calh.reshape(-1,1)),axis=1)
hwah_df = pd.DataFrame(h_val, index=real_df.index, columns=["SIMULATION", "OBSERVED", "CAL_NO_HWAM", "CAL_WITH_HWAH"])

hwah_df = hwah_df.replace({"^2017\d{3}$": pd.np.nan, "-99": pd.np.nan}, regex=True) #drop the error
hwah_df = hwah_df.dropna()

hwah = pd.DataFrame(hwah_df.values.astype(np.int32), index=hwah_df.index, columns=hwah_df.columns)

h_rmse = np.sqrt(mean_squared_error(hwah.iloc[:, 0].values, hwah.iloc[:, 1].values))


"""
190316 visualize the impact of calibration
comparing calh vs real vs sim

visualize the 8 point of data by bar plot
"""
#HWAH
hwah2 = hwah.iloc[:, [0,1,3]]

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1,1,1)
name = ["Koshihikari", "observed", "SD2"]

n = 0
for i in range(len(hwah2.columns)):
    ax.bar(np.arange(len(hwah2.index))+n, hwah2.iloc[:, i], color=cm.hsv(i/3),
           label=name[i], width=0.8/3)
    n = n + 0.8/3
    
plt.legend(loc="best",fontsize=14)
plt.xticks(np.arange(len(hwah2.index)+0.55), ["Kasai_q2_low", "Kasai_q2_high", "Kasai_d4_low",
           "Kasai_d4_high", "Makabe_low", "Makabe_high", "Akita_low", "Akita_high"], 
    rotation=30, fontsize=14)
plt.title("Comparison of yield among observed, koshi_param and SD2_param", fontsize=18)
plt.xlabel("field name", fontsize=15)
plt.ylabel("Yield (kg/ha)", fontsize=15)
plt.ylim([0, 7000])
plt.setp(ax.get_yticklabels(), fontsize=14, visible=True)
plt.savefig("190313_rice_gencalc_png/190316_HWAH_result_SD2.png", bbox_inches="tight")

plt.show()
    

#ADAT
adat2 = adat.iloc[:, [0,1,3]].drop(7)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1,1,1)
name = ["Koshihikari", "observed", "SD2"]

n = 0
for i in range(len(adat2.columns)):
    ax.bar(np.arange(len(adat2.index))+n, adat2.iloc[:, i], color=cm.hsv(i/3),
           label=name[i], width=0.8/3)
    n = n + 0.8/3
    
plt.legend(loc="best",fontsize=14)
plt.xticks(np.arange(len(adat2.index)+0.55), ["Kasai_q2_low", "Kasai_q2_high", "Kasai_d4_low",
           "Kasai_d4_high", "Makabe_low", "Makabe_high", "Akita_low", "Akita_high"], 
    rotation=30, fontsize=14)
plt.title("Comparison of anthesis date among observed, koshi_param and SD2_param", fontsize=18)
plt.xlabel("field name", fontsize=15)
plt.ylabel("Anthesis date", fontsize=15)
plt.ylim([2018200, 2018240])
ax.set_yticklabels(doylist2Date(ax.get_yticks() -2018000))
plt.setp(ax.get_yticklabels(), fontsize=14, rotation=45, visible=True)
plt.savefig("190313_rice_gencalc_png/190316_ADAT_result_SD2.png", bbox_inches="tight")

plt.show()


#MDAT
mdat2 = mdat.iloc[:, [0,1,3]].drop(7)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1,1,1)
name = ["Koshihikari", "observed", "SD2"]

n = 0
for i in range(len(mdat2.columns)):
    ax.bar(np.arange(len(mdat2.index))+n, mdat2.iloc[:, i], color=cm.hsv(i/3),
           label=name[i], width=0.8/3)
    n = n + 0.8/3
    
plt.legend(loc="best",fontsize=14)
plt.xticks(np.arange(len(mdat2.index)+0.55), ["Kasai_q2_low", "Kasai_q2_high", "Kasai_d4_low",
           "Kasai_d4_high", "Makabe_low", "Makabe_high", "Akita_low", "Akita_high"], 
    rotation=30, fontsize=14)
plt.title("Comparison of anthesis date among observed, koshi_param and SD2_param", fontsize=18)
plt.xlabel("field name", fontsize=15)
plt.ylabel("Maturity date", fontsize=15)
plt.ylim([2018230, 2018280])
ax.set_yticklabels(doylist2Date(ax.get_yticks() -2018000))
plt.setp(ax.get_yticklabels(), fontsize=14, rotation=45, visible=True)
plt.savefig("190313_rice_gencalc_png/190316_MDAT_result_SD2.png", bbox_inches="tight")

plt.show()











