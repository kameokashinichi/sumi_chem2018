#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 11:39:39 2018

@author: kameokashinichi
"""

import os
import pickle
import numpy as np
import pandas as pd
import re
import shutil
from statistics import mean, stdev, median
import matplotlib.pyplot as plt
from pylab import cm
from weatherAPI import extractWTHFromDirectory

#extract .wth file from the directory downloaded


#convert wth to pandas Dataframe
def WTH2Dataframe(path, full=False):
    """
    path: str
        the relative path of the directory where the target file is existing
    full: bool
        if full argument is True, the path should be written as hull path    
    """    
    
    record = []
    if full == True:
        with open(path, 'r') as f:
        #print(type(f))
            for row in f:
            #print(type(row))
                record.append(row.strip())
    else:
        with open(os.getcwd() + '/' + path, 'r') as f:
        #print(type(f))
            for row in f:
            #print(type(row))
                record.append(row.strip())        

        
    WTH = record[4:]
    weather = []
    for row in WTH:
        element = row.split()
        weather.append(element)
    
    weather = np.asarray(weather)
    df = pd.DataFrame(weather[1:, 1:].astype(np.float32), index=weather[1:,0].astype(np.int32), columns=weather[0,1:])

    return df

def WTD2DataFrame(path, full=False):
    """
    path: str
        the relative path of the directory where the target file is existing
    full: bool
        if full argument is True, the path should be written as hull path
    
    """    
    WTD = []
    if full == True:
        with open(path, 'r') as f:
        #print(type(f))
            for row in f:
            #print(type(row))
                WTD.append(row.strip())
    else:
        with open(os.getcwd() + '/' + path, 'r') as f:
        #print(type(f))
            for row in f:
            #print(type(row))
                WTD.append(row.strip())        
            
    weather = []
    for row in WTD:
        element = row.split()
        weather.append(element)
    
    weather[0] = weather[0][1:]
    weather = np.asarray(weather)
    ind = []
    for i in range(1, np.shape(weather)[0]):
        ind.append(weather[i, 0][2:])
    #print(weather[0])
    df = pd.DataFrame(weather[1:, 1:].astype(np.float32), index=ind.astype(np.int32), columns=weather[0,1:])

    return df    


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
    


def generateListOfStatistics(dflist, mode='average'):
    """
    dflist : list
        the list which contains 100 dataframes of weather scenario
        
    mode : str
        if mode='average', this function returns the dataframe of each day average
        else, returns standard deviation
    
    """
    
    df = pd.DataFrame(index=dflist[0].index, columns=dflist[0].columns)
    for j in range(len(dflist[0].index)):
        for k in range(len(dflist[0].columns)):
            a = []
            for i in range(len(dflist)):
                a.append(float(dflist[i].iloc[j, k]))
                
            if mode == 'average':
                ave = mean(a)
                df.iloc[j, k] = ave
            else:
                dev = stdev(a)
                df.iloc[j, k] = dev
                
    return df


def visualizeWeather(wthdf, title):
    """
    reference for xtick -> http://python-remrin.hatenadiary.jp/entry/2017/05/27/114816
    
    wthdf : pandas.DataFrame
        the dataframe which contains the information of weather data
        
    title_bn_nn_an : str
        the title of the graph which bn_nn_an condition you use.
    """
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    for i in range(3):
        ax.plot(np.arange(len(wthdf.index)), wthdf.iloc[:, i], color=cm.hsv(i/5), label=wthdf.columns[i])
    
    ax.bar(np.arange(len(wthdf.index)), wthdf.iloc[:, 3], color=cm.hsv(4/5), label=wthdf.columns[3])
    
    print('SRAD : ' + repr(mean(wthdf.loc[:, 'SRAD'])) + ', TMAX : ' + repr(mean(wthdf.loc[:, 'TMAX'])) + 
          ', TMIN : ' + repr(mean(wthdf.loc[:, 'TMIN'])) + ', RAIN : ' + repr(sum(wthdf.loc[:, 'RAIN'])))
    
    plt.legend(loc='best')
    #plt.xticks(np.arange(len(wthdf.index)), np.inspace(1, 365, num=5))
    plt.title(title, fontsize=18)
    plt.xlabel('Day of year', fontsize=16)
    plt.ylabel('temperature(celcius), Solar radiation(MJ/m2), precipitation(mm)', fontsize=12)
    
    plt.show()
    #plt.savefig(figname)
    
    
def visualizeAverageWgenSimulation(id, title):
    wth = extractWTHFromDirectory(id)
    wdflist = []
    for i in range(len(wth)):
        a = WTH2Dataframe(id, wth[i])
        wdflist.append(a)
        
    avedf = generateListOfStatistics(wdflist, mode='average')
    visualizeWeather(avedf, title)



def generateYearList(uruu=None):
    date = []
    for i in range(1, 13):
        if i == 2:
            if uruu == None:
                for j in range(1, 29):
                    a = repr(i) + '/' + repr(j)
                    date.append(a)
            else:
                for j in range(1, 30):
                    a = repr(i) + '/' + repr(j)
                    date.append(a)                
        elif i == 4 or i == 6 or i == 9 or i == 11:
            for j in range(1, 31):
                a = repr(i) + '/' + repr(j)
                date.append(a)
        else:
            for j in range(1, 32):
                a = repr(i) + '/' + repr(j)
                date.append(a)
    return date


def generateDateList(num, year):
    """
    num: int
        the number of doy
    year: int
        the targeted year
    """
    
    lis = []
    for doy in range(1, num+1):
        if year%4 == 0:
            if doy >= 1 and doy< 32:
                date = '1/' + repr(doy)
            elif doy >= 32 and doy < 61:
                date = '2/' + repr(doy - 31)
            elif doy >= 61 and doy < 92:
                date = '3/' + repr(doy - 60)
            elif doy >= 92 and doy < 122:
                date = '4/' + repr(doy - 91)
            elif doy >= 122 and doy < 153:
                date = '5/' + repr(doy - 121)
            elif doy >= 153 and doy < 183:
                date = '6/' + repr(doy - 152)
            elif doy >= 183 and doy < 214:
                date = '7/' + repr(doy - 182)
            elif doy >= 214 and doy < 245:
                date = '8/' + repr(doy - 213)
            elif doy >= 245 and doy < 275:
                date = '9/' + repr(doy - 244)
            elif doy >= 275 and doy < 306:
                date = '10/' + repr(doy - 274)
            elif doy >= 306 and doy < 336:
                date = '11/' + repr(doy - 305)
            elif doy >= 336 and doy < 367:
                date = '12/' + repr(doy - 335)
    
        else:
            if doy >= 1 and doy< 32:
                date = '1/' + repr(doy)
            elif doy >= 32 and doy < 60:
                date = '2/' + repr(doy - 31)
            elif doy >= 60 and doy < 91:
                date = '3/' + repr(doy - 59)
            elif doy >= 91 and doy < 121:
                date = '4/' + repr(doy - 90)
            elif doy >= 121 and doy < 152:
                date = '5/' + repr(doy - 120)
            elif doy >= 152 and doy < 182:
                date = '6/' + repr(doy - 151)
            elif doy >= 182 and doy < 213:
                date = '7/' + repr(doy - 181)
            elif doy >= 213 and doy < 244:
                date = '8/' + repr(doy - 212)
            elif doy >= 244 and doy < 274:
                date = '9/' + repr(doy - 243)
            elif doy >= 274 and doy < 305:
                date = '10/' + repr(doy - 273)
            elif doy >= 305 and doy < 335:
                date = '11/' + repr(doy - 304)
            elif doy >= 335 and doy < 366:
                date = '12/' + repr(doy - 334)
                
        lis.append(date)
        
    return lis
    


#generate monthly forecast
def monthlyWeatherStatistics(id):
    """
    id : weather scenario ID downloaded in working directory
    """
    
    wth = extractWTHFromDirectory(id)
    wdflist = []
    for i in range(len(wth)):
        a = WTH2Dataframe(id, wth[i])
        wdflist.append(a)
        
    year = generateYearList()        
    
    for i in range(len(wdflist)):
        wdflist[i].index = year
        
    mon = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'july', 'aug',
           'sep', 'oct', 'nov', 'dec']
    stat = ['mean','max','min','stdev','median']
    box = np.zeros((len(mon)*5, len(wdflist[0].columns)))
    for i in range(len(mon)):
        m = []
        for k in range(len(wdflist)):
            a = list(filter(lambda x: re.search('^' + repr(i+1) + '/', x), wdflist[0].index))
            b = wdflist[k].loc[a]
            m.append(b)
        mondf = pd.concat(m, axis=0)
        #print(np.shape(mondf))
        for j in range(len(mondf.columns)):
            box[i*5+0, j] = mean(mondf.loc[:, wdflist[0].columns[j]])
            box[i*5+1, j] = max(mondf.loc[:, wdflist[0].columns[j]])
            box[i*5+2, j] = min(mondf.loc[:, wdflist[0].columns[j]]) 
            box[i*5+3, j] = stdev(mondf.loc[:, wdflist[0].columns[j]])
            box[i*5+4, j] = median(mondf.loc[:, wdflist[0].columns[j]])
    
    newindex = []
    for i in range(len(mon)):
        for j in range(len(stat)):
            a = mon[i] + '_' + stat[j]
            newindex.append(a)
            
    df = pd.DataFrame(box, index =newindex, columns = wdflist[0].columns)
    return df


def monthlyWeatherStatsInSingleYear(wdf):
    """
    wdf: pandas.dataframe
        the dataframe for the historical weather data or generated weather
        warning: hte index of the dataframe must be like '18306' (year and doy)
    """
    
    if int(wdf.index[0][:2])%4 == 0:
        year = generateDateList(len(wdf.index), year=4)
    else:
        year = generateDateList(len(wdf.index), year=3)
    wdf.index = year    
    
    lastmon = int(re.search('^\d{1,2}', wdf.index[-1]).group(0))
    mon = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'july', 'aug',
           'sep', 'oct', 'nov', 'dec']
    stat = ['mean','max','min','stdev','median']    
    box = np.zeros((lastmon*5, len(wdf.columns)))
    for i in range(lastmon):
        a = list(filter(lambda x: re.search('^' + repr(i+1) + '/', x), wdf.index))
        b = wdf.loc[a]
        for j in range(len(wdf.columns)):
            box[i*5+0, j] = mean(b.loc[:, wdf.columns[j]])
            box[i*5+1, j] = max(b.loc[:, wdf.columns[j]])
            box[i*5+2, j] = min(b.loc[:, wdf.columns[j]]) 
            box[i*5+3, j] = stdev(b.loc[:, wdf.columns[j]])
            box[i*5+4, j] = median(b.loc[:, wdf.columns[j]])
        
    newindex = []
    for i in range(lastmon):
        for j in range(len(stat)):
            a = mon[i] + '_' + stat[j]
            newindex.append(a)
            
    df = pd.DataFrame(box, index =newindex, columns = wdf.columns)
    return df
    

def monthlyWeatherStatsFromList(wdflist):
    """
    wdflist : list
        the list which contains WTH files in one id
    """        
    
    for i in range(len(wdflist)):
        if int(wdflist[i].index[0][:2])%4 == 0:
            year = generateDateList(len(wdflist[i].index), year=4)
        else:
            year = generateDateList(len(wdflist[i].index), year=3)
        wdflist[i].index = year
    
    #lastmon = re.search('^\d{1,2}', wdflist[i].index).group(0)
    mon = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'july', 'aug',
           'sep', 'oct', 'nov', 'dec']
    stat = ['mean','max','min','stdev','median']
    box = np.zeros((len(mon)*5, len(wdflist[0].columns)))
    for i in range(len(mon)):
        m = []
        for k in range(len(wdflist)):
            a = list(filter(lambda x: re.search('^' + repr(i+1) + '/', x), wdflist[0].index))
            b = wdflist[k].loc[a]
            m.append(b)
        mondf = pd.concat(m, axis=0)
        print(np.shape(mondf))
        for j in range(len(mondf.columns)):
            box[i*5+0, j] = mean(mondf.loc[:, wdflist[0].columns[j]])
            box[i*5+1, j] = max(mondf.loc[:, wdflist[0].columns[j]])
            box[i*5+2, j] = min(mondf.loc[:, wdflist[0].columns[j]]) 
            box[i*5+3, j] = stdev(mondf.loc[:, wdflist[0].columns[j]])
            box[i*5+4, j] = median(mondf.loc[:, wdflist[0].columns[j]])
    
    newindex = []
    for i in range(len(mon)):
        for j in range(len(stat)):
            a = mon[i] + '_' + stat[j]
            newindex.append(a)
            
    df = pd.DataFrame(box, index =newindex, columns = wdflist[0].columns)
    return df

#compare two weather generator data
def visualizeComparedTwoData(statdf1, label1, statdf2, label2, stat, item, unit, save=None):
    
    """
    statdf1 : pandas.Dataframe
        dataframe for monthly statistic data  (ex. pdisagws weather data)
    label1 : str
        the legend title for statdf1 (ex. 'pdisagws')
    statdf2 : pandas.Dataframe
        dataframe for monthly statistic data (ex. kgen weather data)
    label2 : str
        the legend title for statdf2 (ex. 'kgen')
    stat : str
        type of statistics; median, mean, max, min, stdev is defined
    item : str
        type of climate; SRAD, TMAX, TMIN, RAIN is defined
    unit : str
        unit of choosed item; MJ/m2, celcius, mm is used
    
    """    
    
    mon = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'july', 'aug',
           'sep', 'oct', 'nov', 'dec']
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    ax.plot(np.arange(0, len(searchElementforX(statdf1, stat)), 1), 
            searchElementforX(statdf1, stat).loc[:, item], 
            label = label1, color = 'red')
    ax.plot(np.arange(0, len(searchElementforX(statdf2, stat)), 1), 
            searchElementforX(statdf2, stat).loc[:, item], 
            label = label2, color = 'blue')
    
    plt.legend(loc = 'best')
    plt.title(label1 + ' VS ' + label2 +" "+ item + ' ' + stat)
    plt.xlabel('month')
    plt.ylabel(item + '(' + unit + ')')
    plt.xticks(np.arange(0, len(searchElementforX(statdf1, "median")), 1), 
               mon, rotation=30)

    if save:
        plt.savefig('png/' + save)
    plt.show()
    
def visualizeComparedTwoDataWithBar(statdf1, label1, statdf2, label2, stat, item, unit, save=None):
    
    """
    statdf1 : pandas.Dataframe
        dataframe for monthly statistic data  (ex. pdisagws weather data)
    label1 : str
        the legend title for statdf1 (ex. 'pdisagws')
    statdf2 : pandas.Dataframe
        dataframe for monthly statistic data (ex. kgen weather data)
    label2 : str
        the legend title for statdf2 (ex. 'kgen')
    stat : str
        type of statistics; median, mean, max, min, stdev is defined
    item : str
        type of climate; SRAD, TMAX, TMIN, RAIN is defined
    unit : str
        unit of choosed item; MJ/m2, celcius, mm is used
    
    """    
    
    mon = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'july', 'aug',
           'sep', 'oct', 'nov', 'dec']
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    ax.bar(np.arange(0, len(searchElementforX(statdf1, stat)), 1), 
            searchElementforX(statdf1, stat).loc[:, item], width=0.4,
            label = label1, color = 'red', )
    ax.bar(np.arange(0, len(searchElementforX(statdf2, stat)), 1)+0.4, 
            searchElementforX(statdf2, stat).loc[:, item], width=0.4,
            label = label2, color = 'blue')
    
    plt.legend(loc = 'best')
    plt.title(label1 + ' VS ' + label2 +" "+ item + ' ' + stat)
    plt.xlabel('month')
    plt.ylabel(item + '(' + unit + ')')
    plt.xticks(np.arange(0, len(searchElementforX(statdf1, "median")), 1), 
               mon, rotation=30)

    if save:
        plt.savefig('png/' + save, bbox_inches='tight')
    plt.show()    



def multipleStatVisualization(pdisstat, kgenstat, stat, savedate=None):
    
    """
    pdisstat : pandas.Dataframe
        dataframe for monthly statistic data for pdisagws weather data
    kgenstat : pandas.Dataframe
        dataframe for monthly statistic data for kgen weather data    
    stat : str
        type of statistics; median, mean, max, min, stdev is defined
    savedate : str
        the name of the saved figure
    """
    
    fig, axes = plt.subplots(4, 1, sharex=True, figsize = (7, 8))
    mon = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'july', 'aug',
           'sep', 'oct', 'nov', 'dec']
    unit = ['MJ/m2', 'celcius', 'celcius', 'mm']
    for i in range(4):
        axes[i].plot(np.arange(0, len(searchElementforX(pdisstat, stat)), 1), 
                searchElementforX(pdisstat, stat).iloc[:, i], 
                label = 'pdisagws', color = 'red')
        axes[i].plot(np.arange(0, len(searchElementforX(kgenstat, stat)), 1), 
                searchElementforX(kgenstat, stat).iloc[:, i], 
                label = 'kgen', color = 'blue')
    
        axes[i].legend(bbox_to_anchor = (1.1, 0.9))
        axes[i].set_title('KGen VS Pdisagws ' + kgenstat.columns[i] + ' ' + stat)
        axes[i].set_ylabel(kgenstat.columns[i] + ' (' + unit[i] + ')')
    plt.xlabel('month')    
    plt.xticks(np.arange(0, len(searchElementforX(kgenstat, stat)), 1), 
                   mon, rotation=30)
    if savedate:
        plt.savefig('png/' + savedate)    
    plt.show()


def multipleItemVisualization(pdisstat, kgenstat, item, savedate=None):
    
    """
    pdisstat : pandas.Dataframe
        dataframe for monthly statistic data for pdisagws weather data
    kgenstat : pandas.Dataframe
        dataframe for monthly statistic data for kgen weather data    
    item : str
        type of measured value ('SRAD', 'TMAX', 'TMIN', 'RAIN')
    """
    
    fig, axes = plt.subplots(5, 1, sharex=True, figsize = (7, 9))
    mon = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'july', 'aug',
           'sep', 'oct', 'nov', 'dec']
    stat = ['median', 'mean', 'max', 'min', 'stdev']
    unit = {'SRAD':'MJ/m2', 'TMAX':'celcius', 
            'TMIN':'celcius', 'RAIN':'mm'}
    for i in range(len(stat)):
        axes[i].plot(np.arange(0, len(searchElementforX(pdisstat, stat[i])), 1), 
                searchElementforX(pdisstat, stat[i]).loc[:, item], 
                label = 'pdisagws', color = 'red')
        axes[i].plot(np.arange(0, len(searchElementforX(kgenstat, stat[i])), 1), 
                searchElementforX(kgenstat, stat[i]).loc[:, item], 
                label = 'kgen', color = 'blue')
    
        axes[i].legend(bbox_to_anchor = (1.1, 0.9))
        axes[i].set_title('KGen VS Pdisagws ' + item + ' ' + stat[i])
        axes[i].set_ylabel(item + ' (' + unit[item] + ')')
    plt.xlabel('month')    
    plt.xticks(np.arange(0, len(searchElementforX(kgenstat, stat[0])), 1), 
                   mon, rotation=30)
    if savedate:
        plt.savefig('png/' + savedate)
    plt.show()


def compareTwoAnnualWeather(df1, lab1, df2, lab2, item, title):
    """
    df1: pandas.dataframe
        the dataframe of weather data converted from WTH or WTD     
    df2: pandas.dataframe
        the dataframe of weather data converted from WTH or WTD
    item: str
        the type of measured climate; SRAD, TMAX, TMIN, RAIN is defined 
    
    """
    unit = {'SRAD':'MJ/m2', 'TMAX':'celcius', 
            'TMIN':'celcius', 'RAIN':'mm'}
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)    
    ax.plot(np.arange(len(df1.index)), df1.loc[:, item], color=cm.hsv(1/2), label=lab1)
    ax.plot(np.arange(len(df2.index)), df2.loc[:, item], color=cm.hsv(2/2), label=lab2)

    plt.legend(loc='best')
    #plt.xticks(np.arange(len(wthdf.index)), np.inspace(1, 365, num=5))
    plt.title(title, fontsize=18)
    plt.xlabel('Day of year', fontsize=16)
    plt.ylabel(item + '(' + unit[item] + ')', fontsize=12)
    
    plt.show()


def DATE2DOY(datetime):
    """
    datetime: str
        the format of the datetime is like "yyyy-mm-dd"
        if 'Nan' date is assigned as datetime, this function returns 0.

    return doy: int
    """
    if type(datetime) != str or datetime != datetime:
        if type(datetime) == pd._libs.tslibs.timestamps.Timestamp:
            datetime = repr(datetime.year)+"-"+repr(datetime.month).zfill(2)+"-"+repr(datetime.day).zfill(2)
        else:    
            return 0
    elif re.search("^\d{4}.\d{2}.\d{2}$", datetime) == None:
        if re.search("^\d{4}.\d{1,}.\d{1,}$", datetime):
            datetime = re.sub(r"(\D)(\d{1})(($|\D))", r"\g<1>0\g<2>\g<3>", datetime)
        else:
            pass
    else:
        datetime = datetime
        
    date = datetime[5:]
    year = int(datetime[:4])
    DOY = 0
    if year % 4 == 0:
        if date[:2] == '01':
            DOY = DOY + int(date[3:])
        elif date[:2] == '02':
            DOY = 31 + int(date[3:])
        elif date[:2] == '03':
            DOY = 60 + int(date[3:])
        elif date[:2] == '04':
            DOY = 91 + int(date[3:])
        elif date[:2] == '05':
            DOY = 121 + int(date[3:])
        elif date[:2] == '06':
            DOY = 152 + int(date[3:])
        elif date[:2] == '07':
            DOY = 182 + int(date[3:])
        elif date[:2] == '08':
            DOY = 213 + int(date[3:])
        elif date[:2] == '09':
            DOY = 244 + int(date[3:])
        elif date[:2] == '10':
            DOY = 274 + int(date[3:])
        elif date[:2] == '11':
            DOY = 305 + int(date[3:])
        elif date[:2] == '12':
            DOY = 335 + int(date[3:])

    else:
        if date[:2] == '01':
            DOY = DOY + int(date[3:])
        elif date[:2] == '02':
            DOY = 31 + int(date[3:])
        elif date[:2] == '03':
            DOY = 59 + int(date[3:])
        elif date[:2] == '04':
            DOY = 90 + int(date[3:])
        elif date[:2] == '05':
            DOY = 120 + int(date[3:])
        elif date[:2] == '06':
            DOY = 151 + int(date[3:])
        elif date[:2] == '07':
            DOY = 181 + int(date[3:])
        elif date[:2] == '08':
            DOY = 212 + int(date[3:])
        elif date[:2] == '09':
            DOY = 243 + int(date[3:])
        elif date[:2] == '10':
            DOY = 273 + int(date[3:])
        elif date[:2] == '11':
            DOY = 304 + int(date[3:])
        elif date[:2] == '12':
            DOY = 334 + int(date[3:])

    return DOY
    
    
def DOY2DATE(doy, year=2018):
    """
    doy: int
        the day of the year, 1th January -> 1, 31th December -> 365
    year: int
        the year of targeted year (ex. 2018)

    return datetime: str (ex. 2018-01-01)
    """

    if year%4 == 0:
        if doy >= 1 and doy< 32:
            date = repr(year) + '-01-{0:02d}'.format(doy)
        elif doy >= 32 and doy < 61:
            date = repr(year) + '-02-{0:02d}'.format(doy - 31)
        elif doy >= 61 and doy < 92:
            date = repr(year) + '-03-{0:02d}'.format(doy - 60)
        elif doy >= 92 and doy < 122:
            date = repr(year) + '-04-{0:02d}'.format(doy - 91)
        elif doy >= 122 and doy < 153:
            date = repr(year) + '-05-{0:02d}'.format(doy - 121)
        elif doy >= 153 and doy < 183:
            date = repr(year) + '-06-{0:02d}'.format(doy - 152)
        elif doy >= 183 and doy < 214:
            date = repr(year) + '-07-{0:02d}'.format(doy - 182)
        elif doy >= 214 and doy < 245:
            date = repr(year) + '-08-{0:02d}'.format(doy - 213)
        elif doy >= 245 and doy < 275:
            date = repr(year) + '-09-{0:02d}'.format(doy - 244)
        elif doy >= 275 and doy < 306:
            date = repr(year) + '-10-{0:02d}'.format(doy - 274)
        elif doy >= 306 and doy < 336:
            date = repr(year) + '-11-{0:02d}'.format(doy - 305)
        elif doy >= 336 and doy < 367:
            date = repr(year) + '-12-{0:02d}'.format(doy - 335)

    else:
        if doy >= 1 and doy< 32:
            date = repr(year) + '-01-{0:02d}'.format(doy)
        elif doy >= 32 and doy < 60:
            date = repr(year) + '-02-{0:02d}'.format(doy - 31)
        elif doy >= 60 and doy < 91:
            date = repr(year) + '-03-{0:02d}'.format(doy - 59)
        elif doy >= 91 and doy < 121:
            date = repr(year) + '-04-{0:02d}'.format(doy - 90)
        elif doy >= 121 and doy < 152:
            date = repr(year) + '-05-{0:02d}'.format(doy - 120)
        elif doy >= 152 and doy < 182:
            date = repr(year) + '-06-{0:02d}'.format(doy - 151)
        elif doy >= 182 and doy < 213:
            date = repr(year) + '-07-{0:02d}'.format(doy - 181)
        elif doy >= 213 and doy < 244:
            date = repr(year) + '-08-{0:02d}'.format(doy - 212)
        elif doy >= 244 and doy < 274:
            date = repr(year) + '-09-{0:02d}'.format(doy - 243)
        elif doy >= 274 and doy < 305:
            date = repr(year) + '-10-{0:02d}'.format(doy - 273)
        elif doy >= 305 and doy < 335:
            date = repr(year) + '-11-{0:02d}'.format(doy - 304)
        elif doy >= 335 and doy < 366:
            date = repr(year) + '-12-{0:02d}'.format(doy - 334)

    return date



#この関数はインデックスで指定するので、列名検索の時は転置させる必要有り
def searchElementforX(data, *args, strict = False):
       
    csv = data
    
    if strict:

    #ここで欲しい元素データを抽出する
        for j in range(len(args)):        
            b = 0
            for i in range(len(csv.index)):   #データの数（行数分）だけループ
                m = re.fullmatch(args[j], csv.index[i]) #正規表現を用いて、re.fullmatchメソッドで、index名がelementに合致するかを調べる
                if m and b == 0:  #初めて合致した時
                    a = csv.iloc[i]   #その行をaとして保存(実際には列として保存される)
                
                    #print(locals()["a"])   ##デバッグ用の関数。local変数を見る時
                
                    b = b + 1   #すでに一つ合致したことの合図
                elif m and b != 0:   #二回目以降に合致した時
                    c = csv.iloc[i]   #その行をcとして保存（実際には列として保存される）
                    b = b + 1 
                    
                    #print(locals())   #デバッグ用の関数。local変数を見る時
                
                    a = pd.concat([a, c], axis = 1)   #1番最初に保存したaとcを結合し、aを更新。これをある分だけループ。（axis = 1なのは、
                    #一行のみを抽出するとseries型になり、列扱いとなるため）
                else:
                #print(locals()["m"])   ##デバッグ用の関数。local変数を見る時
                
                    continue   #違う元素はスキップ
                    
            if j == 0:
            
                newdf = a
            
            else:
                seconddf = a
                newdf = pd.concat([newdf, seconddf], axis = 1)                    
                    
    else:
        
        for j in range(len(args)):        
            b = 0
            for i in range(len(csv.index)):   #データの数（行数分）だけループ
                m = re.search(args[j], csv.index[i]) #正規表現を用いて、re.fullmatchメソッドで、index名がelementに合致するかを調べる
                if m and b == 0:  #初めて合致した時
                    a = csv.iloc[i]   #その行をaとして保存(実際には列として保存される)
                
                    #print(locals()["a"])   ##デバッグ用の関数。local変数を見る時
                
                    b = b + 1   #すでに一つ合致したことの合図
                elif m and b != 0:   #二回目以降に合致した時
                    c = csv.iloc[i]   #その行をcとして保存（実際には列として保存される）
                    b = b + 1 
                    
                    #print(locals())   #デバッグ用の関数。local変数を見る時
                
                    a = pd.concat([a, c], axis = 1)   #1番最初に保存したaとcを結合し、aを更新。これをある分だけループ。（axis = 1なのは、
                    #一行のみを抽出するとseries型になり、列扱いとなるため）
                else:
                #print(locals()["m"])   ##デバッグ用の関数。local変数を見る時
                
                    continue   #違う元素はスキップ        
               
            if j == 0:
                #print(a)
                newdf = a
            
            else:
                seconddf = a
                newdf = pd.concat([newdf, seconddf], axis = 1)
        
    newdf = newdf.T   #行と列が入れ替わった状態を元に戻す
    
    return newdf








