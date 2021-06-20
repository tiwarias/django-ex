from django.shortcuts import render, redirect
from django.views.generic.edit import CreateView
from django.contrib import messages
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import Group, AbstractUser
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_protect
from django.http import HttpResponse, JsonResponse, HttpResponseRedirect
from django.template.response import TemplateResponse
from django.db import models
from django.utils.decorators import method_decorator
from django.utils.html import escape
import os, requests, uuid, json
import pandas as pd
import numpy as np
import xlwt
import logging
import logging.config
import yaml
import bleach
from django.urls import path
from fuzzywuzzy import fuzz
from .views import *
import spacy
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from collections import Counter
from PIL import Image
from .forms import CreateUserForm

def loginPage(request):        
        if request.method == 'POST':
            username = bleach.clean(request.POST.get('username'))
            password = bleach.clean(request.POST.get('password'))

            user = authenticate(request, username=username, password=password)
            if user:
                login(request, user)
                return HttpResponseRedirect('/')
            else:
                messages.info(request, 'Username Or Password is Incorrect')
                return TemplateResponse(request, 'login.html')
    
        return TemplateResponse(request, 'login.html')

def logoutUser(request):
        logout(request)
        messages.success(request, 'Logged out successfully')
        return HttpResponseRedirect(request,'login.html')

@method_decorator(login_required, name = 'home')
@method_decorator(login_required, name = 'admin_index')
class admin_class():
    
    def __init__(self):
        self.df_dataset = pd.read_csv(r"./BI_Chat_Analysis/Dataset.csv")

    def home(self,requests):
        return TemplateResponse(requests, 'index.html')
        
    def selection(self,requests):
        return TemplateResponse(requests, 'selection.html')
    
    def admin_index(self,requests):
        username = requests.user.username
        if requests.user.is_staff:
            intent_dict_data = self.intent_function()
            cat_dict_data = self.category_function()
            return TemplateResponse(requests, 'admin_index.html', context= {'cat' : cat_dict_data, 'intent' : intent_dict_data})
        else:
            exception_text = "You are not authorized to perform admin functions"
            return TemplateResponse(requests, 'index.html', context = {'exception_text': exception_text})
        
        
    def category_function(self):
        list_category = []
        for c_l in self.df_dataset.loc[:,'Category']:
            list_category.append(c_l)
        list_category = list(set(list_category))
        cat_dict = dict(zip(range(len(list_category)),list_category))
        return cat_dict
       
    
    def intent_function(self):
       
        df2 = pd.read_csv(r"./BI_Chat_Analysis/Dataset_intent.csv")
        list_intent=[]
        for i_l in df2.loc[:,'Intent']:
            list_intent.append(i_l)
        list_intent = list(set(list_intent))
        intent_dict = dict(zip(range(len(list_intent)),list_intent))
        return intent_dict
    
    
    def synonym_del(self,request):
        intent_dict_data = self.intent_function()
        cat_dict_data = self.category_function()
        if request.method == "POST":
            category = request.POST.get('category_id')
            intent = request.POST.get("intent_input")
            keyword = request.POST.get("keyword_input")
        
        
        ratio = []
        cat_intent = ""
        
        for i in range(len(self.df_dataset)):
            val = fuzz.token_sort_ratio(keyword.lower(), self.df_dataset['Keyword'].iloc[i].lower())
            ratio.append(val)
            if val > 90:
                if self.df_dataset['Category'].iloc[i] == category:
                    print(self.df_dataset.index[i], self.df_dataset.Keyword[i])
                    try:
                        self.df_dataset.drop(self.df_dataset.index[i], inplace=True)
                        delete_dtime = pd.datetime.now()
                        df_del_key = pd.DataFrame({'Keyword':[keyword],'Category':[category],'Timestamp':[delete_dtime]})
                        
                        df_del_key.to_csv('./BI_Chat_Analysis/Deleted_keywords.csv',mode='a',index = False,header=None)
                        exception_text = "Keyword deleted successfully"
                        break
                    except:
                        logging.error('Deleting keyword from an open file')
                        exception_text = "Please close the file before deleting the keyword"
                        return TemplateResponse(request, "admin_index.html", context = {'cat':cat_dict_data, 'int' : intent_dict_data, 'exception_text': exception_text})
                else:
                    continue
        if max(ratio)<60:
            
            exception_text = "Selected Keyword does not exist"
        
        self.df_dataset.reset_index(drop=True, inplace=True)
        try:
            self.df_dataset.to_csv('./BI_Chat_Analysis/Dataset.csv', index=False)
        except:
            logging.warning('Could not save changes to the file')
            
        return TemplateResponse(request, "admin_index.html", context = {'cat':cat_dict_data, 'int' : intent_dict_data, 'exception_text': exception_text})
     
                     
    def fetch_keywords(self,request):
        if request.method == "POST":
            cat_dropdown = request.POST.get('category_id')
    
        
        df_dropdown = self.df_dataset
        print(self.df_dataset)
        df_dropdown = df_dropdown.loc[self.df_dataset['Category'] == cat_dropdown]
        df_dropdown = df_dropdown['Keyword']
        df_dropdown.reset_index(drop=True, inplace=True)
        
    
        keyword_list = df_dropdown.tolist()
        intent_dict_data = self.intent_function()
        cat_dict_data = self.category_function()
        
        exception_text = ""
        keyword_dict = dict(zip(range(len(keyword_list)), keyword_list))
       
    
        return TemplateResponse(request, "admin_index.html", {'cat' : cat_dict_data, 'keyword_list': keyword_dict,'cat_dropdown_val':cat_dropdown})
        
    
    def synonym_add(self,request):
        intent_dict_data = self.intent_function()
        cat_dict_data = self.category_function()
        if request.method == "POST":
            category = request.POST.get('category_id')
            intent = request.POST.get("intent_input")
            keyword = request.POST.get("keyword_input")
            
            if keyword == "":
                exception_message = "Please enter a keyword"
                return TemplateResponse(request, "admin_index.html", context={'cat':cat_dict_data, 'int' : intent_dict_data, 'exception_message': exception_message})
        
        ratio = []
        cat_intent = ""
        
        for i in range(len(self.df_dataset)):
            val = fuzz.token_sort_ratio(keyword.lower(), self.df_dataset['Keyword'].iloc[i].lower())
            ratio.append(val)
            print(val)
            if val > 95:
                exception_text = "Keyword already exists for selected category"
                break
        if max(ratio)<95:
            try:
                print(ratio)
                updated_dtime = pd.datetime.now()
                self.df_dataset.loc[len(self.df_dataset)] = [keyword,category,updated_dtime]
                exception_text = "Keyword added successfully"
                logging.info('Keyword added successfully')
                self.df_dataset.to_csv('./BI_Chat_Analysis/Dataset.csv', index=False)
            except:
                
                logging.error('Writing to an open file')
                exception_text = "Please close the file before updating"
                return TemplateResponse(request, "admin_index.html", context={'cat':cat_dict_data, 'int' : intent_dict_data, 'exception_text': exception_text})
                
        return TemplateResponse(request, "admin_index.html", context={'cat':cat_dict_data, 'int' : intent_dict_data, 'exception_text': exception_text})
        

def main_admin():
    obj_admin = admin_class()
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename="../app.log", format=LOG_FORMAT, level=logging.ERROR)
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f.read())
        f.close()
        logging.config.dictConfig(config)
    return obj_admin