#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 15:21:31 2019

@author: kameokashinichi
"""

import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
import math
from statistics import mean, stdev
import matplotlib.cm as cm
import requests
import subprocess
import json

def dataframe2wtd(df, out_name, add_tave=True, to_slice=False, to_check=True, out_extension = ".WTDE"):
    check_string = ["DAY", "SRAD", "TMAX", "TMIN", "RAIN"]
    check = np.isin(check_string, df.columns)
    if not(np.all(check)) and to_check:
        raise ValueError("A field is missing in the input dataframe. \
                         DAY, SRAD, TMAX, TMIN, RAIN are all mandatory")
    if out_name[-len(out_extension):] == out_extension:
        out_name = out_name[:-len(out_extension)]
    df = df.copy()
    
    if not(np.isin("YEAR", df.columns)):
        is_biss = (max(df["DAY"]) == 366)
        df.insert(0, "YEAR", 2019 + is_biss)
    df["DATE"]= df["YEAR"] * 1000 + df["DAY"]
    columns = ["DATE", "SRAD", "TMAX", "TMIN", "RAIN"]
    print(df.dtypes)
    header = "@  DATE  SRAD  TMAX  TMIN  RAIN"
    fmt = "%03d% 6.1f% 6.1f% 6.1f% 6.1f"
    if add_tave:
        if not(np.isin("TAVE", df.columns)):
            tavg_ary = (df["TMIN"] + df["TMAX"]) / 2
            df.insert(4, "TAVE", tavg_ary)
        columns += ["TAVE"]
        header += "  TAVE"
        fmt += "% 6.1f"
    df_ary = df[columns].values
    if to_slice:
        slice_indexes = (df["DAY"]==df["DAY"][0])&(df["YEAR"]==df["YEAR"][0])
        slice_indexes = np.arange(len(df))[slice_indexes]
        slice_indexes = slice_indexes
        for i in range(len(slice_indexes) - 1):
            np.savetxt("{}{:04d}{}".format(out_name, i+1, out_extension),
               df_ary[slice_indexes[i]:slice_indexes[i+1]],
               fmt=fmt, header=header, comments="")
        np.savetxt("{}{:04d}{}".format(out_name, len(slice_indexes), out_extension),
            df_ary[slice_indexes[-1]:],
            fmt=fmt, header=header, comments="")
        return ["{}{:04d}{}".format(out_name, i+1, out_extension) for i in range(len(slice_indexes))]
    else:
        np.savetxt("{}{}".format(out_name,out_extension),
            df_ary, fmt=fmt, header=header, comments="")
        return "{}{}".format(out_name,out_extension)


"""
190226_check the data of Sumitomo chemical for calibration by GenCalc

the item of the data frame
['year', 'prefecture', 'field', 'latitude', 'longitude', 'variety',
       'cultivation_method', 'planting_density_per_m2', 'row_spacing_cm',
       'planting_density_per_3.3_m2', 'basal_nitrogen_kg_per_10a',
       'additional_nitrogen_kg_per_10a_1st', 'plant_number_per_hill',
       'sowing_date', 'transplanting_date', 'panicle_formation_date',
       'heading_date', 'proper_time_for_harvesting',
       'actual_brown_rice_weight_after_moisture_15_percentage_correction_kg_per_10a',
       'brown_rice_weight_after_moisture_15_percentage_correction_kg_per_10a',
       'grain_filling_ratio',
       'grain_weight_above_1.8mm_sieve_per_under_1.8mm_sieve_g_per_15hills',
       '1000_grain_weight_after_moisture_15_percentage_correction_and_pound_rice_percentage_correction_g',
       'panicle_number_per_m2', 'lodging_index', 'culm_length_cm',
       'panicle_length_cm', 'harvest_date', 'grain_number_per_m2',
       'moisture_percentage', 'amylose_percentage',
       'grain_weight_above_1.85mm_sieve_g_per_15hills_2018',
       'grain_weight_above_1.8mm_sieve_g_per_15hills',
       'grain_weight_above_1.8mm_sieve_g_per_15hills_2018',
       'grain_weight_above_1.9mm_sieve_g_per_15hills',
       'grain_weight_above_2.0mm_sieve_g_per_15hills',
       'grain_weight_above_2.1mm_sieve_g_per_15hills',
       'grain_weight_above_2.2mm_sieve_g_per_15hills',
       'grain_weight_under_1.8mm_sieve_g_per_15hills', '1st_node_cm',
       '2nd_node_cm', '3rd_node_cm', '4th_node_cm', '5th_node_cm',
       '6th_node_cm', '®±%', '¢n±%', 'íQ±%', 'Ó±%', 'Ä%',
       '
F±%', '·±%', 'û%', 'î%', ' %', 'Â¢n%', '»Ì¼%'],
      dtype='object')

The data which can be used is
['year', 'prefecture', 'field', 'latitude', 'longitude', 'variety', 
 'sowing_date', 'transplanting_date', 'panicle_formation_date',
 'heading_date', 'proper_time_for_harvesting', 'basal_nitrogen_kg_per_10a',
 'actual_brown_rice_weight_after_moisture_15_percentage_correction_kg_per_10a']

1. extract the necessary information from data source "2015_2018_low_density_yield.csv"
2. change the name of columns to DSSAT parameter
3. convert the data to .RIA file
"""

csv = pd.read_csv("2015_2018_low_density_yield.csv", encoding="latin1", index_col=0, header=0)

csv2 = csv.loc[:, ['year', 'prefecture', 'field', 'latitude', 'longitude', 'variety', 
                  'sowing_date', 'transplanting_date', 'panicle_formation_date',
                  'heading_date', 'proper_time_for_harvesting', 'basal_nitrogen_kg_per_10a',
                  'actual_brown_rice_weight_after_moisture_15_percentage_correction_kg_per_10a']]

csv2.columns = ['year', 'prefecture', 'field', 'latitude', 'longitude', 'variety', 
 'sowing_date', 'transplanting_date', 'IDAT', 'ADAT', 'MDAT', 'basal_nitrogen_kg_per_10a',
 'HWAM']


def genCalibrationfileFromDf(df, out_name, crop="rice"):
    """
    df   : pandas.DataFrame
        dataframe which contains crop growth result.
    out_name: str
        the name of the A file for crop calibration.
    crop : str
        the name of the targeted crop.
    """
    
    if crop == "rice":
        parlist = ["IDAT", "ADAT", "MDAT", "SWAH", "PWAM", "CWAM", "LAIX", "HWAM"]
    elif crop == "wheat":
        parlist = ["IDAT","ADAT","MDAT","HWAM","L#SM","T#AM","H#AM","HWUM"]
    else:
        parlist = ["ADAT","MDAT","CWAM","HWAM","LAIX","PWAM","H#AM","HWUM"]
        
    
    temp = """*EXP. DATA (A): JPRI0001.RI
"""
    param = []
    for i in df.columns.values:
        if any([x == i for x in parlist]):
            param.append(i)
    
    headerNum = ""
    for i in range(len(param)-1):
        headerNum = headerNum + "%6s"
    
    header = """\
%5s%7s{par}
""".format(par=headerNum)
    #print(header)
    
    #header = header % ("@TRNO", *param,)  #unpack the list in order to let the element become argument
    
    f_val = ""
    n = 0
    
    for i in range(len(df.index)):
        
        par_dat = list(filter(lambda x: re.search("DAT", x), param))
        par = list(filter(lambda x: re.search("DAT", x)==None, param))
        d_lis = df.loc[df.index[i], par_dat].values
        v_lis = df.loc[df.index[i], par].values
        if len(v_lis) <= 1:
            j = 0
            try:
                if v_lis[j] != v_lis[j] or v_lis[j] == "NaT" or v_lis[j] == None:
                    v_lis[j] = "-99" 
                elif type(v_lis[j]) != str:
                    if v_lis[j] > 100:
                        v_lis[j] = int(v_lis[j])
                    else:
                        v_lis[j] = round(v_lis[j], 2)
                elif re.search("^\d+\.?\d*$", v_lis[j]) == None:
                    v_lis[j] = "-99"
                elif len(v_lis[j]) > 6:
                    v_lis[j] = v_lis[j][:5]                    
            except:
                print(i, param[j], type(v_lis[j]))
                
        else:
            try:
                for j in range(len(v_lis)):
                    if v_lis[j] != v_lis[j] or v_lis[j] == "NaT" or v_lis[j] == None:
                        v_lis[j] = "-99" 
                    elif type(v_lis[j]) != str:
                        if v_lis[j] > 100:
                            v_lis[j] = int(v_lis[j])
                        else:
                            v_lis[j] = round(v_lis[j], 2)
                    elif re.search("^\d+\.?\d*$", v_lis[j]) == None:
                        v_lis[j] = "-99"
                    elif len(v_lis[j]) > 6:
                        v_lis[j] = v_lis[j][:5]                    
            except:
                print(i, param[j], type(v_lis[j]))
                
        if len(d_lis) <= 1:
            j = 0
            try:
                if d_lis[j] != d_lis[j] or d_lis[j] == "NaT" or d_lis[j] == None:
                    d_lis[j] = "-99" 
                elif type(d_lis[j]) != str:
                    if d_lis[j] > 10000:
                        d_lis[j] = repr(int(d_lis[j]))[2:]
                    else:
                        pass                   
                elif re.search("^\d+\.?\d*$", d_lis[j]) == None:
                    d_lis[j] = "-99"
                elif float(d_lis[j]) >= 10000:
                    d_lis[j] = d_lis[j][2:]
                else:
                    pass
            except:
                print(i, param[j], type(d_lis[j]))
                
        else:
            try:
                for j in range(len(d_lis)):
                    if d_lis[j] != d_lis[j] or d_lis[j] == "NaT" or d_lis[j] == None:
                        d_lis[j] = "-99" 
                    elif type(d_lis[j]) != str:
                        if d_lis[j] > 10000:
                            d_lis[j] = repr(int(d_lis[j]))[2:]
                        else:
                            pass                   
                    elif re.search("^\d+\.?\d*$", d_lis[j]) == None:
                        d_lis[j] = "-99"
                    elif float(d_lis[j]) >= 10000:
                        d_lis[j] = d_lis[j][2:]
                    else:
                        pass                   
            except:
                print(i, param[j], type(d_lis[j]))
        
        if type(v_lis) != list:
            v_lis = list(v_lis)
        if type(d_lis) != list:
            d_lis = list(d_lis)            
        v_lis = v_lis + d_lis
        
        if all([x == "-99" for x in v_lis]):
            pass
        else:
            n = n+1
            val = setValues(param)
            #print(val)
            val = val % (i+1, *v_lis,)
            f_val = f_val + val
    
    if type(par) != list:
        par = list(par)
    if type(par_dat) != list:
        par_dat = list(par_dat)            
    par = par + par_dat
    
    header = header % ("@TRNO", *par,) 
    
    #return header
    with open(out_name, 'w') as f:
        f.write(temp+header+f_val)


def setValues(param):
    
    headerNum = ""
    for i in range(len(param)):
        if i == 0:
            headerNum = headerNum + "%7s"
        else:
            headerNum = headerNum + "%6s"
            
    
    val = """%5.0f{par}\
""".format(par=headerNum)
    
    return val


   


    















