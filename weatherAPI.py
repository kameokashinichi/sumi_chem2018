#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 15:20:32 2018

@author: kameokashinichi
"""

import os
import pickle
import numpy as np
import pandas as pd
import re
import shutil
import requests
import json

"""
send_from_directory(directory, filename) function in application.py is to let 
directory(specified by directory) download link for file(specified by filename).

url_for('function name') function is to specify the url for defined function.

"""

#get Weather scenario ID by using get_scenarios() function in application.py
def getScenarioId_dev():
    
    os.chdir("/Users/kameokashinichi/Documents/postdoc/listenfield/weatherAPI")
    
    url = "http://dev.listenfield.com:8080/weather/generator/v1.1/scenarios"
    r = requests.get(url)
    idlist = []
    for i in range(len(r.text.split(','))):
        m = re.search("\d.{1,}\w", r.text.split(',')[i])
        idlist.append(m.group())
    return idlist

def getScenarioId_production():
    
    os.chdir("/Users/kameokashinichi/Documents/postdoc/listenfield/weatherAPI")
    
    url = "http://ec2-52-196-202-21.ap-northeast-1.compute.amazonaws.com/weather/generator/v1.1/scenarios"
    r = requests.get(url)
    idlist = []
    for i in range(len(r.text.split(','))):
        m = re.search("\d.{1,}\w", r.text.split(',')[i])
        idlist.append(m.group())
    return idlist

def getPdisagwsId():
    
    os.chdir("/Users/kameokashinichi/Documents/postdoc/listenfield/weatherAPI")
    
    url = "http://ec2-52-196-202-21.ap-northeast-1.compute.amazonaws.com/weather/generator/v1.1/scenarios"
    r = requests.get(url)
    idlist = []
    for i in range(len(r.text.split(','))):
        m = re.search("\d.{1,}\w", r.text.split(',')[i])
        idlist.append(m.group())
    return idlist



#get specified weather scenario by using get_scenario(id)
def getScenarioFromId(id, type="json"):
    
    url = "http://dev.listenfield.com:8080/weather/generator/v1.1/scenarios/" + id
    
    r = requests.get(url)
    info = r.text
    infojson = json.loads(info)
    if type == "json":
        return infojson
    else:
        return info

def getScenarioFromId_production(id, type="json"):
    
    url = "http://ec2-52-196-202-21.ap-northeast-1.compute.amazonaws.com/weather/generator/v1.1/scenarios/"+id
    #print(url)
    r = requests.get(url)
    info = r.text
    infojson = json.loads(info)
    if type == "json":
        return infojson
    else:
        return info


    
#print keys of scenario dictionary
def printScenarioKeys(id, production=None):
    if production == None:
        a = getScenarioFromId(id, type="json")
    else:
        a = getScenarioFromId_production(id, type="json")
    for key in a.keys():
        try:
            print(key + ' : ' + repr(a[key].keys()))
        except AttributeError:
            print(key + ' : ')
    #print(a)
    
#Check the scenario parameter used (BN:NN:AN, for example)   
def getScenarioCondition(id, appear=False, production=None):
    if production == None:
        a = getScenarioFromId(id, type="json")
    else:
        a = getScenarioFromId_production(id, type="json")
    param = a['WeatherGeneratorCondition']
    if appear == True:
        print('BN:NN:AN : ' + repr(param['WeatherForecast']) + '\n' + 'model : ' + repr(param['model']))
    return param


def getAndOpenWeatherZip(id, production=None):
    """   
    getScenarioFromId(id, type="json")['result']['downloadLink'] indicates 
    the url of the weather data zip file.
    
    id : str
        the id of the targeted weather scenario
    """  
    
    from urllib import request
    if production == None:
        url = getScenarioFromId(id, type="json")['result']['downloadLink']
    else:
        url = getScenarioFromId_production(id, type="json")['result']['downloadLink']
    request.urlretrieve(url, id + '.zip')
    
    import zipfile
     
    with zipfile.ZipFile(id + '.zip','r') as inputFile:
        inputFile.extractall()
    
    os.remove(id + '.zip')
    

def getPngFromUrl(url, id):
    """
    create png file from download URL(url) to current working directory
    
    """
    from urllib import request
    request.urlretrieve(url, id + '.png')    
    
    
#return dictionary which contains scenario ID sorted by weather model    
def getSucceedScenarioIdDict():
    succeed = {'KGen':[], 'pdisagws':[]}
    for i in range(len(getScenarioId())):
        try:
            senario = getScenarioCondition(getScenarioId()[i], appear=False)
            if senario['model'] == 'KGen':
                succeed['KGen'].append(getScenarioId()[i])
            else:
                succeed['pdisagws'].append(getScenarioId()[i])
        except KeyError:
            pass
            #print('model could not generate in ' + repr(i) + 'th')
    return succeed

def getSucceedScenarioIdDictFromList(idlist, pdisag=None):
    succeed = {'KGen':[], 'pdisagws':[]}
    for i in range(len(idlist)):
        try:
            if pdisag == None:
                senario = getScenarioCondition(idlist[i], appear=False)
            else:
                senario = getScenarioCondition(idlist[i], appear=False, pdisag='pdisag')
            if senario['model'] == 'KGen':
                succeed['KGen'].append(idlist[i])
            else:
                succeed['pdisagws'].append(idlist[i])
        except KeyError:
            pass
            #print('model could not generate in ' + repr(i) + 'th')
    return succeed


#get weather scenario ID of kgen
def getKGenResultIdList():
    succeed = getSucceedScenarioIdDict()
    kgenresult = succeed['KGen']
    return kgenresult


#for argument kgen, we need to apecify the list of weather scenario ID
def generateKgenWthDictionary(kgenIDlist):
    kgenwthdict = {}
    for i in range(len(kgenIDlist)):
        os.chdir('/Users/kameokashinichi/Documents/postdoc/listenfield/weatherAPI')
        m = filter(lambda x : re.search(kgenIDlist[i], x), os.listdir('.'))
        if len(list(m)) > 0:
            kgenwthdict.update({kgenIDlist[i]:extractWTHFromDirectory(kgenIDlist[i])})
        else:
            kgenwthdict.update({kgenIDlist[i]:[]})
    return kgenwthdict


def extractWTHFromDirectory(id):
    files = os.listdir(os.getcwd() + '/' + id)
    WTH = list(filter(lambda x : re.search('.WTH', x), files))
    return WTH


def generateNewKgenScenario(scenario_num):
    """
    Kgen weather scenario for Agri Sasamoto applying seasonal forecast
    
    bn_nn_an : str
        probability ratio which weather will be (generally "33:34:33")
    model : str
        the weather generator model(currently, 'kgen' or 'pdisagws')
    scenario_num : int
        number of the generated weather (for kgen, 500 is default)
    """
    
    url = "http://dev.listenfield.com:8080/weather/generator/v1.1/scenarios"
    
    param = { 
      "wth_src" : "naro1km", 
      "wgen_model": "kgen",
      "scenario_num" : repr(scenario_num),
      "latitude" : "35.706179",
      "longitude" : "140.482362",
      "from_date" : "2018-01-01",
      "to_date" : "2018-12-31",
      "bn_nn_an" : "33:34:33",
      'jma_forecast_data': {'monthly_forecasts': [{'b_date': '2018-09-01',
                            'e_date': '2018-09-30',
                            'type': 'average_air_temperature',
                            'probabilities': [8, 38, 54]},
                           {'b_date': '2018-09-01',
                            'e_date': '2018-09-30',
                            'type': 'precipitation',
                            'probabilities': [39, 36, 25]},
                           {'b_date': '2018-10-01',
                            'e_date': '2018-10-31',
                            'type': 'average_air_temperature',
                            'probabilities': [4, 32, 64]},
                           {'b_date': '2018-10-01',
                            'e_date': '2018-10-31',
                            'type': 'precipitation',
                            'probabilities': [30, 32, 38]},
                           {'b_date': '2018-11-01',
                            'e_date': '2018-11-30',
                            'type': 'average_air_temperature',
                            'probabilities': [28, 24, 48]},
                           {'b_date': '2018-11-01',
                            'e_date': '2018-11-30',
                            'type': 'precipitation',
                            'probabilities': [23, 33, 44]}]},
                              "monthly_adjust": True,
                              "snow_adjust": False
    }
    headers = {
    'Content-Type': "application/json",
    'Cache-Control': "no-cache",
    'Postman-Token': "45ab4e0a-77e0-483d-9494-5f69124a3ba6"
    }
    
    payload = json.dumps(param)
    
    response = requests.request("POST", url, data=payload, headers=headers)
    return response


def generateKgenScenario(scenario_num):
    """
    Kgen weather scenario for Agri Sasamoto
    
    bn_nn_an : str
        probability ratio which weather will be (generally "33:34:33")
    model : str
        the weather generator model(currently, 'kgen' or 'pdisagws')
    scenario_num : int
        number of the generated weather (for kgen, 500 is default)
    """
    url = "http://dev.listenfield.com:8080/weather/generator/v1.1/scenarios"
    
    param = { 
      "wth_src" : "naro1km", 
      "wgen_model": "kgen",
      "scenario_num" : repr(scenario_num),
      "latitude" : "35.706179",
      "longitude" : "140.482362",
      "from_date" : "2018-01-01",
      "to_date" : "2018-12-31",
      "bn_nn_an" : '33:34:33', 
      "monthly_adjust": True,
      "snow_adjust": False
    }
    headers = {
        'Content-Type': "application/json",
        'Cache-Control': "no-cache",
        'Postman-Token': "bafe6574-aa67-4a5f-9ca9-11c9266b017f"
        }
    
    payload = json.dumps(param)
    
    response = requests.request("POST", url, data=payload, headers=headers)
    return response
    

def updateScenarios(id, from_date, to_date):
    
    url = "http://dev.listenfield.com:8080/weather/generator/v1.1/scenarios/" + id
    
    payload = { 
              "mode": "update",
              "wth_src" : "naro1km",
              "from_date" : from_date,
              "to_date" : to_date 
            }
    
    headers = {
        'Content-Type': "application/json",
        'Cache-Control': "no-cache",
        'Postman-Token': "bafe6574-aa67-4a5f-9ca9-11c9266b017f"
        }
    
    response = requests.request("PUT", url, data=payload, headers=headers)
    
    return response.text


def getPlotUrl(id):
    url = getScenarioFromId(id)['plot']
    url = re.sub('^\.', 'http://dev.listenfield.com:8080/weather', url)
    return url


def getHistoricalData(lat, lon, start, end, title=None):
    """
    lat: float
        the value of the latitude of the targeted place.
    lon: float
        the value of the longitude of the targeted place.
    start: str ('yyyy-mm-dd')
        the starting date of the request
    end: str ('yyyy-mm-dd')
        the ending date of the request
    title: str
        the title for the wtd file
    """
    
    url = "http://ec2-52-196-202-21.ap-northeast-1.compute.amazonaws.com/weather/v1/data"
    
    querystring = {"latitude":repr(lat),"longitude":repr(lon),
                   "from_date":start,"to_date":end,
                   "wth_src":"naro1km","out_fmt":"wtd"}
    
    headers = {
        'Cache-Control': "no-cache",
        'Postman-Token': "9c163359-00c4-4932-95b7-98f549712024"
        }
    
    response = requests.request("GET", url, headers=headers, params=querystring)
    
    wtd = []
    for row in response.text.split('\n'):
        wtd.append(row)
        
    if title:
        with open(title + '.wtd', 'w') as f:
            for i in wtd:
                f.write(i + '\n')
    
    return wtd
    
"""
for i in range(1980, 2017):
    getHistoricalData(35.706179, 140.482362, repr(i)+'-01-01',repr(i)+'-12-31', 'sasa'+repr(i))
    
"""



    
    
    
    
    
    
    