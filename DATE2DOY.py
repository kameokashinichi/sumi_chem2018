 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 08:45:44 2018

@author: kameokashinichi
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re


#Change datetime to DOY
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
    if type(doy) == np.float64:
        doy = int(doy)
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


def datelist2DOY(datelist):
    """
    datelist: list or numpy ndarray
        the list for the date 
        
    return doylist: list
    """
    
    doylist = []
    for i in datelist:
        doy = DATE2DOY(i)
        doylist.append(doy)
    
    return doylist

def doylist2Date(doylist, year=2018):
    datelist = []
    for i in doylist:
        #print(i)
        date = DOY2DATE(i, year=year)
        datelist.append(date)
    
    return datelist

def prepareDfDict(keylist, csvlist):
    """
    keylist: list (ex. month = ['march', 'april', 'may', 'june', 'july', 'august'])
        the list for the keywords which can identify the csv.
    csvlist: list (ex. list(filter(lambda x: re.search('\.csv$', x), os.listdir()))
        the list for the path of the csv file.
        
    return dic: dict
        the dictionary which is composed of keyword and pandas Dataframe
    """
    
    dic = dict()
    for i in keylist:
        for j in csvlist:
            if re.search(i, j) != None:
                file = pd.read_csv('cropsim_csv/evaluate_forecast_impact/' + j, header=0)
                dic.update({i:file})
    return dic


def prepareDfDict_new(keylist, csvlist):
    """
    keylist: list (ex. month = ['march', 'april', 'may', 'june', 'july', 'august'])
        the list for the keywords which can identify the csv.
    csvlist: list (ex. list(filter(lambda x: re.search('\.csv$', x), os.listdir()))
        the list for the path of the csv file.
        
    return dic: dict
        the dictionary which is composed of keyword and pandas Dataframe
    """
    
    dic = dict()
    for i in keylist:
        for j in csvlist:
            if re.search(i, j) != None:
                file = pd.read_csv('simriw_result_with_forecast_Newparam/summary/' + j, header=0)
                dic.update({i:file})
    return dic


def prepareDfDict_new2(keylist, csvlist):
    """
    keylist: list (ex. month = ['march', 'april', 'may', 'june', 'july', 'august'])
        the list for the keywords which can identify the csv.
    csvlist: list (ex. list(filter(lambda x: re.search('\.csv$', x), os.listdir()))
        the list for the path of the csv file.
        
    return dic: dict
        the dictionary which is composed of keyword and pandas Dataframe
    """
    
    dic = dict()
    for i in keylist:
        for j in csvlist:
            if re.search(i, j) != None:
                file = pd.read_csv('simriw_c_result_general_default_param/summary/' + j, header=0)
                dic.update({i:file})
    return dic


def prepareDOYDB(dic, target='anthesis'):
    """
    dic: dict 
        the dictionary which is composed of keyword and pandas Dataframe
    target: str
        the target phenological date, 'anthesis' or 'maturity'
        
    return doydb: pandas.Dataframe
        the dataframe of the date for chosen target('anthesis' or 'maturity')
        the header of the dataframe is key of the input dictionary
    """
    
    doy = []
    for i in dic.keys():
        if target == 'anthesis':
            lis = dic[i]['anthesis_date (yyyy-mm-dd)'].values
            lis = datelist2DOY(lis)
            doy.append(lis)
        elif target == 'maturity':
            lis = dic[i]['physiologic_maturity_dat (yyyy-mm-dd)'].values
            lis = datelist2DOY(lis)
            doy.append(lis)
            
    doy = np.asarray(doy)
    doydb = pd.DataFrame(doy.T, columns=dic.keys())
    return doydb


def prepareDOYDB_new(dic, target='anthesis'):
    """
    dic: dict 
        the dictionary which is composed of keyword and pandas Dataframe
    target: str
        the target phenological date, 'anthesis' or 'maturity'
        
    return doydb: pandas.Dataframe
        the dataframe of the date for chosen target('anthesis' or 'maturity')
        the header of the dataframe is key of the input dictionary
    """
    
    doy = []
    for i in dic.keys():
        if target == 'anthesis':
            lis = dic[i]['flowering_date'].values
            lis = datelist2DOY(lis)
            doy.append(lis)
        elif target == 'maturity':
            lis = dic[i]['maturity_dates'].values
            lis = datelist2DOY(lis)
            doy.append(lis)
            
    doy = np.asarray(doy)
    doydb = pd.DataFrame(doy.T, columns=dic.keys())
    return doydb    


def visualizeDATE(DOYdb, title, save=None):
    """
    DOYdb: pandas dataframe (output of prepareDOYDB)
        the dataframe of the date for chosen target('anthesis' or 'maturity')
    title: str
        the title for the boxplot
    save: str (ex. '180903_anthesis_with_3mon')
        if you want to save the figure, dedired file name will be added as save variable
        
    return boxplot for phenological date
    """
    
    fig, ax = plt.subplots()
    ax.boxplot(DOYdb.values)
    plt.ylim([np.unique(DOYdb.values)[1]-5, np.unique(DOYdb.values)[len(np.unique(DOYdb.values))-1]+5])
    ax.set_xticklabels(DOYdb.columns)
    #ax.set_yticklabels(doylist2Date(np.arange(np.unique(DOYdb.values)[1]-5, np.unique(DOYdb.values)[len(np.unique(DOYdb.values))-1]+5, 5)))
    plt.yticks(np.arange(np.unique(DOYdb.values)[1]-5, np.unique(DOYdb.values)[len(np.unique(DOYdb.values))-1]+5, 5), doylist2Date(np.arange(np.unique(DOYdb.values)[1]-5, np.unique(DOYdb.values)[len(np.unique(DOYdb.values))-1]+5, 5)))
    ax.set_title(title)
    if save:
        plt.savefig('png/' + save + '.png', bbox_inches='tight')    
    plt.show()


def compareTwoDf(DOYdb1, DOYdb2, title, save=None):
    """
    DOYdb1: pandas dataframe (output of prepareDOYDB)
        the dataframe of the date for chosen target('anthesis' or 'maturity')
    DOYdb2: pandas dataframe (output of prepareDOYDB)
        the dataframe of the date for chosen target('anthesis' or 'maturity')
    title: str
        the title for the boxplot
    save: str (ex. '180903_anthesis_with_3mon')
        if you want to save the figure, dedired file name will be added as save variable
        
    return boxplot for phenological date
    """
    
    newdf = pd.concat([DOYdb1.iloc[:,0], DOYdb2.iloc[:,0]], axis=1)
    for i in range(1, len(DOYdb1.columns)):    
        newdf = pd.concat([newdf, DOYdb1.iloc[:,i], DOYdb2.iloc[:,i]], axis=1)    

    fig, ax = plt.subplots()
    ax.boxplot(newdf.values)
    plt.ylim([np.unique(newdf.values)[1]-5, np.unique(newdf.values)[len(np.unique(newdf.values))-1]+5])
    ax.set_xticklabels(newdf.columns)
    #ax.set_yticklabels(doylist2Date(np.arange(np.unique(DOYdb.values)[1]-5, np.unique(DOYdb.values)[len(np.unique(DOYdb.values))-1]+5, 5)))
    plt.yticks(np.arange(np.unique(newdf.values)[1]-5, np.unique(newdf.values)[len(np.unique(newdf.values))-1]+5, 5), doylist2Date(np.arange(np.unique(newdf.values)[1]-5, np.unique(newdf.values)[len(np.unique(newdf.values))-1]+5, 5)))
    ax.set_title(title)
    if save:
        plt.savefig('png/' + save + '.png', bbox_inches='tight')    
    plt.show()


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

def printColumnNumber(df, colname):
    [print(i) if df.columns[i] == colname else i for i in range(len(df.columns))]

def returnColumnNumber(df, colname):
    for i in range(len(df.columns)):
        if df.columns[i] == colname:
            return i

            
            
            
            
            
            
            
            
            
            










