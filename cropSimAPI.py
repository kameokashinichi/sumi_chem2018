#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 10:25:35 2018

@author: kameokashinichi
"""

import requests
import re
import json
from urllib import request
import zipfile
import os
import shutil

def getCropSimId():

    url = "http://ec2-52-196-202-21.ap-northeast-1.compute.amazonaws.com/cropsim/v1.1/simulations"
    
    headers = {
        'Cache-Control': "no-cache",
        'Postman-Token': "cbc3aa3c-ac71-436e-97a1-a7c8e29fdc4a"
        }
    
    response = requests.request("GET", url, headers=headers)
    
    #print(response.text)
    
    lis = []
    for row in response.text.split(','):
        m = re.search('\d{4}-\d{2}-\w{5}-\d{2}-\d{2}-\w{1,}', row)
        lis.append(m.group(0))
            
    return lis
    

def getCropSimFromId(id):
    """
    id: str
        the id of the crop simulation
    """
    
    url = "http://ec2-52-196-202-21.ap-northeast-1.compute.amazonaws.com/cropsim/v1.1/simulations/" + id 

    headers = {
        'Cache-Control': "no-cache",
        'Postman-Token': "bf0ca3d5-f686-45e9-8f3e-35a09f93414a"
        }
    
    response = requests.request("GET", url, headers=headers)
    
    jinfo = json.loads(response.text)
    
    return jinfo


def getCsvResultFromId(id):
    """
    id: str
        the id of the crop simulation
    """
    
    url = "http://ec2-52-196-202-21.ap-northeast-1.compute.amazonaws.com/cropsim/v1.1/simulations/" + id 

    headers = {
        'Cache-Control': "no-cache",
        'Postman-Token': "bf0ca3d5-f686-45e9-8f3e-35a09f93414a"
        }
    
    response = requests.request("GET", url, headers=headers)
    
    jinfo = json.loads(response.text)
    csvlink = jinfo['csv_output']
    
    request.urlretrieve(csvlink, 'csv/' + id + '.csv')
    
    #with zipfile.ZipFile(id + '.zip', 'r') as inputFile:
        #inputFile.extractall()
        

def upload_weather_file(weather_zip):
    """
    weather_zip: str
        the name of the weather zip file
    """
    
    url = "http://dev.listenfield.com:3232/cropsim/v1.1/uploads"
    payload = "------WebKitFormBoundary7MA4YWxkTrZu0gW\r\nContent-Disposition: form-data; name=\"upload_file\"; filename="+weather_zip+"\r\nContent-Type: application/zip\r\n\r\n\r\n------WebKitFormBoundary7MA4YWxkTrZu0gW--"
    headers = {
    'content-type': "multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW",
    'Cache-Control': "no-cache",
    'Postman-Token': "6cdee0c7-2f4d-4e09-a60a-d4427083201a"
    }

    response = requests.request("POST", url, data=payload, headers=headers)
    
    print(response.text)


def generateWeatherScenarioID(year, lat="35.706179", lon="140.482362", s_num='10'):
    """
    lat: str
        the value of the latitude for targeted field
    lon: str
        the value of the longitude for targeted field 
    year: int
        the value of the simulated year
    s_num: str
        the number of the weather scenario
        
    return ID: str
        the ID for the weather scenario    
    """
    
    url = "http://ec2-52-196-202-21.ap-northeast-1.compute.amazonaws.com/weather/generator/v1.1/scenarios"
    
    payload ={ 
      "wth_src" : "naro1km", 
      "wgen_model": "kgen",
      "scenario_num" : s_num,
      "latitude" : lat,
      "longitude" : lon,
      "from_date" : repr(year)+"-01-01",
      "to_date" : repr(year)+"-12-31",
      "bn_nn_an" : "33:34:33",
      "monthly_adjust": "true",
      "snow_adjust": "false"
    }
    
    headers = {
    'Content-Type': "application/json",
    'Cache-Control': "no-cache",
    'Postman-Token': "21aa674f-ac54-4c57-98ee-2b291b59c0d9"
    }
    
    payload = json.dumps(payload)

    response = requests.request("POST", url, data=payload, headers=headers)
    
    rj = json.loads(response.text)
    
    return(rj["ID"])
    

def updateWeatherScenario(id, from_date, to_date):
    """
    id: str
        the id of the weather scenario.
    from_date: str
        the start date for updating.
    to_date: str
        the end date for updating.
    """
    
    url = "http://ec2-52-196-202-21.ap-northeast-1.compute.amazonaws.com/weather/generator/v1.1/scenarios/"+id
    payload = { 
      "mode": "update",
      "wth_src" : "naro1km",
      "from_date" : from_date,
      "to_date" : to_date  
    }    
    headers = {
        'Content-Type': "application/json",
        'Cache-Control': "no-cache",
        'Postman-Token': "cd1f60ff-4227-4b3f-89c7-aeb3b02c6450"
        }
    payload = json.dumps(payload)
    
    response = requests.request("PUT", url, data=payload, headers=headers)
    
    print(response.text)
    

def generateCropScenario(transplant_date, weather_id, folder, lat=35.706179, lon=140.482362, upload=False):
    """
    transplant_date: str (ex. 'yyyy-mm-dd')
        the transplanting date for simulation
    weather_id: str
        the ID of weather scenario 
    lat: float
        the latitude of the targeted field
    lon: float
        the longitude of the targeted field
        
    download result csv file
    """
    url = "http://ec2-52-196-202-21.ap-northeast-1.compute.amazonaws.com/cropsim/v1.1/simulations"
    
    if upload == False:
        payload = {
          "transplant_date": transplant_date,
          "crop_ident_ICASA": "RIC",
          "cultivar_name": "Koshihikari",
          "model": "simriw",
          "weather_file": "http://ec2-52-196-202-21.ap-northeast-1.compute.amazonaws.com/weather/static/ex/"+weather_id+".zip",
          "field_latitude": lat,
          "field_longitude": lon,
          "wait": "true",
          "model_params": 
          {
          "tp_cool": "false"
          }
        }
    else:
        payload = {
          "transplant_date": transplant_date,
          "crop_ident_ICASA": "RIC",
          "cultivar_name": "Koshihikari",
          "model": "simriw",
          "upload_file": "http://ec2-52-196-202-21.ap-northeast-1.compute.amazonaws.com/cropsim/uploads/"+weather_id+".zip",
          "field_latitude": lat,
          "field_longitude": lon,
          "wait": "true",
          "model_params": 
          {
          "tp_cool": "false"
          }
        }        
        
    headers = {
        'Content-Type': "application/json",
        'Cache-Control': "no-cache",
        'Postman-Token': "20f2fff0-028d-442d-a730-5c3229eafb78"
        }
    
    payload = json.dumps(payload)
    
    response = requests.request("POST", url, data=payload, headers=headers)

    jinfo = json.loads(response.text)
    csvlink = jinfo['csv_output']
    
    #print(type(csvlink))
    request.urlretrieve(csvlink, folder + "/" + jinfo["id"] + '.csv')















