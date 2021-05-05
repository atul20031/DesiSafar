#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import argparse
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import CountVectorizer ,TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
# ! pip install rank_bm25
import nltk 
import string 
import re 
import nltk
import os.path
from os import path
from IPython.display import Image
from IPython.display import display
from sklearn.metrics.pairwise import sigmoid_kernel
# !pip install pywebio
from pywebio.input import *
from pywebio.output import *
import pywebio.session
from pywebio import start_server
import pickle
import sqlite3
import time
from pywebio.platform.flask import webio_view
from pywebio import STATIC_PATH
from flask import Flask, send_from_directory
nltk.download('wordnet')

from rank_bm25 import BM25Okapi
dataset2 = pd.read_csv('wiki_voyage_correct.csv')
dataset1 = pd.read_csv('final_dataset_wo_duplicates.csv')
conn = sqlite3.connect('test2.db')
conn.execute('''CREATE TABLE IF NOT EXISTS user_details
             (User_id INTEGER PRIMARY KEY     AUTOINCREMENT,
             Name           TEXT    NOT NULL,
             Age            INT     NOT NULL,
             State        CHAR(50)  NOT NULL,
             Pin CHAR(50));''')
main_document_scores=[]
def word2vec_sim_score(q,doc_list):
  if len(doc_list) ==287:
    doc_list = doc_list[:-1]
  doc_list.append(q)
  stemmed_doc_list, doc_vector = list(), list()
  for doc in doc_list:
    stemmed_doc_list.append(doc.split(" "))
  # print(stemmed_doc_list)
  word2vec = Word2Vec(stemmed_doc_list,min_count=1,vector_size=1000)
  for index,row in enumerate(stemmed_doc_list):
    model_vector = (np.mean([word2vec[token] for token in row], axis=0)).tolist()
    doc_vector.append(model_vector)
  sim = cosine_similarity(doc_vector)
  return np.asarray(sim[-1][:-1])

def tf_idf_sim_score(q,doc_list):
  tfvec = TfidfVectorizer(stop_words='english')
  if len(doc_list) ==287:
    doc_list = doc_list[:-1]
  doc_list.append(q)
  vec_corpus = tfvec.fit_transform(doc_list)
  sim = cosine_similarity(vec_corpus)
  return np.asarray(sim[-1][:-1])

def countvec_sim_score(q,doc_list):
  countvec = CountVectorizer(stop_words='english')
  if len(doc_list) ==287:
    doc_list = doc_list[:-1]
  doc_list.append(q)
  vec_corpus = countvec.fit_transform(doc_list)
  sim = cosine_similarity(vec_corpus)
  return np.asarray(sim[-1][:-1])

document_list=[]
lemmatizer = WordNetLemmatizer() 
i=0
cities = dataset2['City']
try:
    with open('document_list.pkl', 'rb') as file:
      
    # Call load method to deserialze
        document_list = pickle.load(file)

except:
    for i in range(dataset2.shape[0]):
      k= cities[i]
      document = ' description of '+ k + dataset2.iloc[i]['description']+'Places to visit in '+k  +str(dataset2.iloc[i]['sites']) + '\n how to reach '+ k+str(dataset2.iloc[i]['how to reach']) + '\n best time to visit'+k+dataset2.iloc[i]['best to time visit'] + ' '+dataset2.iloc[i]['wiki voyage']
      document = document.lower()
      input_str = document
      translator = str.maketrans('', '', string.punctuation) 
      document = document.translate(translator) 
      document=lemmatizer.lemmatize(document)

      document_list.append(document)
      i+=1


    with open('document_list.pkl', 'wb') as file:

        # A new file will be created
        pickle.dump(document_list, file)

total_tags = []
total_tags.append(dataset1['historical & heritage'].value_counts())
total_tags.append(dataset1['city'].value_counts())
total_tags.append(dataset1['pilgrimage'].value_counts())
tags_hills = dataset1['hill station'].value_counts()
total_tags.append(tags_hills)
tags_beach = dataset1['beach'].value_counts()
total_tags.append(tags_beach)
tags_lakes = dataset1['lake & backwater'].value_counts()
total_tags.append(tags_lakes)
tags_adventure = dataset1['adventure / trekking'].value_counts()
total_tags.append(tags_adventure)
tags_wildlife = dataset1['wildlife'].value_counts()
total_tags.append(tags_wildlife)
tags_waterfall = dataset1['waterfall'].value_counts()
total_tags.append(tags_waterfall)
tags_nature = dataset1['nature & scenic'].value_counts()
total_tags.append(tags_nature)


def choices():
#     popup('DesiSafar - A Travel Recommendation System', 'Information Retrieval Project [Group Number 4] \n\nRishabh Bafna (MT20118) \nAtul Rawat(MT20___) \nDivisha Bisht (MT20___) \n Aman Dapola (MT20___) \n Vineet Maheshwari (MT20___) \n\n Special thanks to Professor Rajiv Ratn Shah!')
    
    clear();
    try:
        clear('BTV')
    except:
        pass
    img = open('Images/DesiSafar Logo.jpg', 'rb').read()
    put_image(img, width='900px')
    put_markdown('# **IR Project - Group Number 4**')
    answer = radio("Choose one", options=['Explore Incredible India!', 'Get Travel Recommendations'])
    if(answer == 'Explore Incredible India!'):
        fun()
    if(answer == 'Get Travel Recommendations'):
        put_text('\nLet\'s get started! ')
        select_recommendation_system()

def fun():
    put_markdown('## Please wait! Your request is being processed!')
#     pywebio.session.hold()
    description = dataset2['description']
    count =0
    put_processbar('bar')
    for i in range(1, 11):
        set_processbar('bar', i / 10)
        time.sleep(0.1)
    for place in cities:
        put_html('<hr>')
        put_markdown("# *`%s`*" % place)
        pic = 'Images/' + str(place) + '.jpg'
        img = open(pic, 'rb').read()
        put_image(img, width='1500px')
        desc = description[count]
        desc = desc.strip()
        desc = desc.replace('-','')
        put_text(desc)
        count+=1
        #put_image(img) to get the original size
    #style(put_text('In case of copyright issues, please drop an email to rishabh20118@iiitd.ac.in'), 'color:red')
    put_markdown("# *In case of copyright issues, please drop an email to `rishabh20118@iiitd.ac.in`*")
    img = open('Images/India_1.jpg', 'rb').read()
    put_image(img, width='1500px')
    
def select_recommendation_system():
    recommendation_system = select('Which type of recommendation system would you prefer?', ['BM25 based Recommendation System', 'TF-IDF(Content based)','Count-Vectorizer(Content Based)','Word2Vec(Content Based)','Collaborative Filtering'])
    #BM25 based Recommendation System
    if(recommendation_system == 'BM25 based Recommendation System'):
        put_text('BM25 based Recommendation System is a free text based recommendation system.')
        free_text = textarea('Enter a free text', rows = 3, placeholder = 'Write anything...\n\'snow winter nature trekking\' ... \'lake boating waterfall tiger\' ... \'market clothes nights history\' ... \'beach cruise camping boats ships\' ... \'temples hills altitude winter line\' ... \'lion safari forests camping nature\' ... ')
        query = free_text
        corpus = document_list
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        query=query.lower()
        query=lemmatizer.lemmatize(query)
        put_text('Query after preprocessing '+ query)
        tokenized_query = query.split(" ")
        doc_scores = bm25.get_scores(tokenized_query)
        put_text(query)
        display_recommendations(doc_scores)
        
        
    if(recommendation_system == 'TF-IDF(Content based)'):
        put_text('Content Based recommendation system.')
        free_text = textarea('Enter a free text', rows = 3, placeholder = 'Write anything...\n\'snow winter nature trekking\' ... \'lake boating waterfall tiger\' ... \'market clothes nights history\' ... \'beach cruise camping boats ships\' ... \'temples hills altitude winter line\' ... \'lion safari forests camping nature\' ... ')
        query = free_text
        corpus = document_list
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        query=query.lower()
        query=lemmatizer.lemmatize(query)
        put_text('Query after preprocessing '+ query)
        tokenized_query = query.split(" ")
        doc_scores = tf_idf_sim_score(query,document_list)
        put_text(query)
        display_recommendations(doc_scores)
        
    if(recommendation_system == 'Count-Vectorizer(Content Based)'):
        put_text('Content Based recommendation system.')
        free_text = textarea('Enter a free text', rows = 3, placeholder = 'Write anything...\n\'snow winter nature trekking\' ... \'lake boating waterfall tiger\' ... \'market clothes nights history\' ... \'beach cruise camping boats ships\' ... \'temples hills altitude winter line\' ... \'lion safari forests camping nature\' ... ')
        query = free_text
        corpus = document_list
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        query=query.lower()
        query=lemmatizer.lemmatize(query)
        put_text('Query after preprocessing '+ query)
        tokenized_query = query.split(" ")
        doc_scores = countvec_sim_score(query,document_list)
        put_text(query)
        display_recommendations(doc_scores)  
        
    if(recommendation_system == 'Word2Vec(Content Based)'):
        put_text('Content Based recommendation system.')
        free_text = textarea('Enter a free text', rows = 3, placeholder = 'Write anything...\n\'snow winter nature trekking\' ... \'lake boating waterfall tiger\' ... \'market clothes nights history\' ... \'beach cruise camping boats ships\' ... \'temples hills altitude winter line\' ... \'lion safari forests camping nature\' ... ')
        query = free_text
        corpus = document_list
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        query=query.lower()
        query=lemmatizer.lemmatize(query)
        put_text('Query after preprocessing '+ query)
        tokenized_query = query.split(" ")
        try:
            doc_scores = word2vec_sim_score(query,document_list)
        except Exception as ex:
            put_text(ex)
        put_text(query)
        try:
            display_recommendations(doc_scores)   
        except Exception as ex:
            put_text(ex)
    if(recommendation_system == 'Collaborative Filtering')  :
        account_status = select('Log in or sign up?', ['Log in','Sign Up','Home'])
        if(account_status == 'Log in'):
            log_in_page()
        if(account_status == 'Sign Up'):
            put_text('Sign up selected')
            sign_up_page()
        if(account_status== 'Home'):
            choices()
            
def check_form(data):  # input group validation: return (input name, error msg) when validation fail
        if len(data['name']) > 20:
            return ('name', 'Name too long!')
        if data['age'] <= 0:
            return ('age', 'Age can not be negative!')     
        if data['age'] >= 130:
            return ('age','Enter valid age')
        
def check_log_in(data):
    conn = sqlite3.connect('test2.db')
    user_id = data['user_id']
    pin = data['password']
    cursor = conn.execute('Select Pin from user_details where user_id=?', (user_id,))
    found=0
    for row in cursor:
      if (row[0] == pin):
        put_text('Successful LOGIN')
        found = 1
        
      else:
        put_text('Wrong PIN or account number')
        found = 0
        return('user_id','Invalid Details')
    if(found==0):
        return('user_id','Invalid Details')
    
def insert_details(info):
    conn = sqlite3.connect('test2.db')
    logf = open("insert.log", "w")
#     put_text('Inside insert details',info)
    name = str(info['name'])
    age = int(info['age'])
    pin = str(info['password'])
    state = str(info['state'])
    try:
        conn.execute("INSERT INTO user_details (Name,Age, State,Pin) VALUES (?, ?, ?, ?)",
                             (name, age, state,pin))
        conn.commit()
        cursor = conn.execute("SELECT * FROM user_details ORDER BY user_id DESC LIMIT 1")
        put_text('Insert successful')
    except Exception as ex:
        put_text(ex)
        logf.write("Failed to insert {0}:"+ex)
    try:
        for row in cursor:
            put_text('Please note your User ID:  ', row[0])
            account_number=int(row[0])
    except Exception as ex:
        put_text(ex)
    popup('Please note your u=User ID : '+str(account_number))
    put_buttons(['Home'], onclick=[choices])
    put_buttons(['Explore'], onclick=[select_recommendation_system])
    
    
def log_in_page():
    clear()
    img = open('Images/DesiSafar Logo.jpg', 'rb').read()
    put_image(img, width='900px')
    put_markdown('# **Log In**')
    put_buttons(['Home'], onclick=[choices])
    put_buttons(['Explore'], onclick=[select_recommendation_system])
    info = input_group("User info",[
  input('User Id', name='user_id',type = NUMBER,required=True),
  
        input('Password', type=PASSWORD, name='password', required=True)
       
],validate=check_log_in)
    pywebio.session.hold()

    
def sign_up_page():
    clear()
    img = open('Images/DesiSafar Logo.jpg', 'rb').read()
    put_image(img, width='900px')
    put_markdown('# **Sign UP**')
#     data = input_group("Basic info", [
#         input('Input your name', name='name'),
#         input('Input your age', name='age', type=NUMBER)
#     ], validate=check_form)
#     put_markdown("`data = %r`" % data)
        
    info = input_group("User info",[
  input('Name', name='name',required=True),
  input('Input your age', name='age', type=NUMBER,required=True,),
        input('password', type=PASSWORD, name='password', required=True),
        select('Select your state',["Andhra Pradesh","Arunachal Pradesh ","Assam","Bihar","Chhattisgarh","Goa","Gujarat","Haryana","Himachal Pradesh","Jammu and Kashmir","Jharkhand","Karnataka","Kerala","Madhya Pradesh","Maharashtra","Manipur","Meghalaya","Mizoram","Nagaland","Odisha","Punjab","Rajasthan","Sikkim","Tamil Nadu","Telangana","Tripura","Uttar Pradesh","Uttarakhand","West Bengal","Andaman and Nicobar Islands","Chandigarh","Dadra and Nagar Haveli","Daman and Diu","Lakshadweep","National Capital Territory of Delhi","Puducherry"]
,name='state',required=True)
],validate=check_form)
    
#     put_markdown("`info = %r`" % info)
    
#     put_text(info['name'],info['age'],info['password'],info['state'])
    insert_details(info)
    pywebio.session.hold()
            
        
def display_recommendations(document_scores):
    clear();
    global main_document_scores
    try:
        main_document_scores = document_scores.tolist()
        put_text(''+str(len(main_document_scores)))
    except Exception as ex:
        put_text(ex)
   
    try:
        clear('BTV')
    except:
        pass
    img = open('Images/DesiSafar Logo.jpg', 'rb').read()
    put_image(img, width='900px')
    put_markdown('# **IR Project - Group Number 4**')
    recommendations = int(input('Enter number of recommendations you want : '))
    ds= document_scores
    recommendations_index =ds.argsort()[-recommendations:][::-1]
    print(recommendations_index)
    cities = dataset2['City']
    desc = dataset2['description']
    
    displayed_recommendations = []
    displayed_recommendations_index ={}
    print()
    for i in range(recommendations_index.shape[0]):
      put_html('<hr>')
      put_html('<hr>')
      pic = 'Images/' + str(cities[recommendations_index[i]]) + '.jpg'
      img = open(pic, 'rb').read()
      put_image(img, width='1500px')
      put_markdown("# *`%s`*" % cities[recommendations_index[i]])
      t=desc[recommendations_index[i]]
      t = t.strip()
      t = t.replace('-','')
      put_text(t)
      displayed_recommendations.append(cities[recommendations_index[i]])
      displayed_recommendations_index[cities[recommendations_index[i]]]=recommendations_index[i]
      print('-------------------------------------------------------')
    try:
        selected_recommendation = select('Explore :', displayed_recommendations)
        display_details(displayed_recommendations_index[selected_recommendation])
    except Exception as ex:
        put_text(ex)
    
def display_recommendations_temp():
    put_text('Displayin more recommendations.....'+str(len(main_document_scores)))
    try:
        doc_score = np.array(main_document_scores)
        display_recommendations(doc_score)
    except Exception as ex:
        put_text(ex)
    put_text(''+str(doc_score.shape[0]))
def display_details(selected_index):
    row = dataset2.iloc[selected_index]
    clear()
    put_markdown('# **IR Project - Group Number 4**')
    put_markdown("# *`%s`*" % cities[selected_index])
    put_html('<hr>')
    pic = 'Images/' + str(cities[selected_index]) + '.jpg'
    img = open(pic, 'rb').read()
    put_image(img, width='1500px')
    
    put_markdown(r""" #  Description""", lstrip=True)
    t = row['description']
    t = t.strip()
    t = t.replace('-','')
    put_text(t)
    try:
        
#         put_buttons([('Best time to visit',str(selected_index)), ('How to reach',str(selected_index))], onclick=[btv, htr])
#         put_buttons([dict(label='Best time to visit', value=str(selected_index), color='primary')],onclick = btv)
#         put_buttons([dict(label='How to reach', value=str(selected_index), color='primary')],onclick = htr)
#         put_buttons([dict(label='Places to visit', value=str(selected_index), color='primary')],onclick = ptv)
        put_grid([
                [put_buttons([dict(label='Best time to visit', value=str(selected_index), color='primary')],onclick = btv),
                 put_buttons([dict(label='How to reach', value=str(selected_index), color='primary')],onclick = htr),
               put_buttons([dict(label='Places to visit', value=str(selected_index), color='primary')],onclick = ptv),
                 put_buttons(['Back'], onclick=[display_recommendations_temp]),
                put_buttons(['Home'], onclick=[choices])],
                
               
            ], cell_width='150px', cell_height='100px')
        pywebio.session.hold()
        
    except Exception as ex:
        put_text(ex)
    

    
def btv(selected_index):
    selected_index = int(selected_index)
    try:
        set_scope('BTV',-1,-1,'clear')
        clear('BTV')
        row = dataset2.iloc[selected_index]
        put_markdown(r""" #  Best time to visit""", lstrip=True,scope='BTV')
        t = row['best to time visit']
        t = t.strip()
        t = t.replace('-','')
        put_text(t,scope='BTV')
    except Exception as ex:
        put_text(ex)
        
def ptv(selected_index):
    selected_index = int(selected_index)
    try:
        set_scope('BTV',-1,-1,'clear')
        clear('BTV')
        row = dataset2.iloc[selected_index]
        put_markdown(r""" #  Near by places to visit""", lstrip=True,scope='BTV')
        t = row['sites']
        t = t.strip()
        t = t.replace('-','')
        put_text(t,scope='BTV')
    except Exception as ex:
        put_text(ex)
    
def htr(selected_index):
    selected_index = int(selected_index)
    try:
        row = dataset2.iloc[selected_index]
        set_scope('BTV',-1,-1,'clear')
        clear('BTV')
        put_markdown(r""" #  How to reach""", lstrip=True,scope='BTV')
        t = row['how to reach']
        t = t.strip()
        t = t.replace('-','')
        put_text(t,scope='BTV')
    except Exception as ex:
        put_text(ex)

app = Flask(__name__)
# app.add_url_rule('/', 'webio_view', webio_view(choices), methods=['GET', 'POST', 'OPTIONS'])
# app.run()

app.add_url_rule('/tool', 'webio_view', webio_view(choices),
            methods=['GET', 'POST', 'OPTIONS'])
# app.run(host='localhost', port=80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=8080)
    args = parser.parse_args()

    start_server(choices, port=args.port)
# if __name__ == '__main__':
#     choices()


# In[24]:


# from rank_bm25 import BM25Okapi
# dataset2 = pd.read_csv('wiki_voyage_correct.csv')
# dataset1 = pd.read_csv('final_dataset_wo_duplicates.csv')
# main_document_scores=[]
# def word2vec_sim_score(q,doc_list):
#   if len(doc_list) ==287:
#     doc_list = doc_list[:-1]
#   doc_list.append(q)
#   stemmed_doc_list, doc_vector = list(), list()
#   for doc in doc_list:
#     stemmed_doc_list.append(doc.split(" "))
#   # print(stemmed_doc_list)
#   word2vec = Word2Vec(stemmed_doc_list,min_count=1,vector_size=1000)
#   for index,row in enumerate(stemmed_doc_list):
#     model_vector = (np.mean([word2vec[token] for token in row], axis=0)).tolist()
#     doc_vector.append(model_vector)
#   sim = cosine_similarity(doc_vector)
#   return np.asarray(sim[-1][:-1])

# def tf_idf_sim_score(q,doc_list):
#   tfvec = TfidfVectorizer(stop_words='english')
#   if len(doc_list) ==287:
#     doc_list = doc_list[:-1]
#   doc_list.append(q)
#   vec_corpus = tfvec.fit_transform(doc_list)
#   sim = cosine_similarity(vec_corpus)
#   return np.asarray(sim[-1][:-1])

# def countvec_sim_score(q,doc_list):
#   countvec = CountVectorizer(stop_words='english')
#   if len(doc_list) ==287:
#     doc_list = doc_list[:-1]
#   doc_list.append(q)
#   vec_corpus = countvec.fit_transform(doc_list)
#   sim = cosine_similarity(vec_corpus)
#   return np.asarray(sim[-1][:-1])

# document_list=[]
# lemmatizer = WordNetLemmatizer() 
# i=0
# cities = dataset2['City']
# for i in range(dataset2.shape[0]):
#   k= cities[i]
#   document = ' description of '+ k + dataset2.iloc[i]['description']+'Places to visit in '+k  +str(dataset2.iloc[i]['sites']) + '\n how to reach '+ k+str(dataset2.iloc[i]['how to reach']) + '\n best time to visit'+k+dataset2.iloc[i]['best to time visit'] + ' '+dataset2.iloc[i]['wiki voyage']
#   document = document.lower()
#   input_str = document
#   translator = str.maketrans('', '', string.punctuation) 
#   document = document.translate(translator) 
#   document=lemmatizer.lemmatize(document)

#   document_list.append(document)
#   i+=1

# total_tags = []
# total_tags.append(dataset1['historical & heritage'].value_counts())
# total_tags.append(dataset1['city'].value_counts())
# total_tags.append(dataset1['pilgrimage'].value_counts())
# tags_hills = dataset1['hill station'].value_counts()
# total_tags.append(tags_hills)
# tags_beach = dataset1['beach'].value_counts()
# total_tags.append(tags_beach)
# tags_lakes = dataset1['lake & backwater'].value_counts()
# total_tags.append(tags_lakes)
# tags_adventure = dataset1['adventure / trekking'].value_counts()
# total_tags.append(tags_adventure)
# tags_wildlife = dataset1['wildlife'].value_counts()
# total_tags.append(tags_wildlife)
# tags_waterfall = dataset1['waterfall'].value_counts()
# total_tags.append(tags_waterfall)
# tags_nature = dataset1['nature & scenic'].value_counts()
# total_tags.append(tags_nature)


# In[4]:



# query = input('Enter query related to tourism : ')

# corpus = document_list
# tokenized_corpus = [doc.split(" ") for doc in corpus]
# bm25 = BM25Okapi(tokenized_corpus)
# query=query.lower()
# query=lemmatizer.lemmatize(query)
# print(query)

# #-------Change here--------
# recommender = int(input('Enter the recommender to use \n 1. BM25 \t 2.TF-IDF+Cosine similarity \n 3.Count Vectorizer + Cosine Similarity \t 4. Word2Vec + Cosine Similarity \t'))
# if(recommender == 1):
#   tokenized_query = query.split(" ")
#   print('Using BM25')
#   doc_scores = bm25.get_scores(tokenized_query)
# if(recommender == 2):
#   score = tf_idf_sim_score(query,document_list)
#   print('Using TF-IDF')
#   doc_scores = score
# if(recommender == 3):
#   score = countvec_sim_score(query,document_list)
#   print('Using Count Vec')
#   doc_scores = score
# if(recommender == 4):
#   print('Using Word2Vec')
#   doc_scores = word2vec_sim_score(query,document_list)

# print('-----------------------------------------------------------------------------')
# #------------------------------

# a= doc_scores
# recommendations = int(input('Enter number of recommendations you want : '))
# recommendations_index =a.argsort()[-recommendations:][::-1]
# cities = dataset2['City']
# desc = dataset2['description']
# print('ID','     ','Place',   '    Description')
# print()
# for i in range(recommendations_index.shape[0]):
#   print(recommendations_index[i],'  ', cities[recommendations_index[i]], ' : ',desc[recommendations_index[i]])
#   print('-------------------------------------------------------')

# selected_recommendation = int(input('\n Enter the id of place you are interested in :'))
# row = dataset2.iloc[selected_recommendation]
# print('-------------------------------------------------------')

# print(' \n Place Description :',row['description'])
# print()
# print('-------------------------------------------------------')

# print('Near by places to visit :\n',row['sites'])
# print()
# print('-------------------------------------------------------')

# print(' How to reach :',row['how to reach'])
# print()
# print('-------------------------------------------------------')

# print('Best time to visit',row['best to time visit'])


# In[65]:


# def choices():
# #     popup('DesiSafar - A Travel Recommendation System', 'Information Retrieval Project [Group Number 4] \n\nRishabh Bafna (MT20118) \nAtul Rawat(MT20___) \nDivisha Bisht (MT20___) \n Aman Dapola (MT20___) \n Vineet Maheshwari (MT20___) \n\n Special thanks to Professor Rajiv Ratn Shah!')
    
#     clear();
#     try:
#         clear('BTV')
#     except:
#         pass
#     img = open('Images/DesiSafar Logo.jpg', 'rb').read()
#     put_image(img, width='900px')
#     put_markdown('# **IR Project - Group Number 4**')
#     answer = radio("Choose one", options=['Explore Incredible India!', 'Get Travel Recommendations'])
#     if(answer == 'Explore Incredible India!'):
#         fun()
#     if(answer == 'Get Travel Recommendations'):
#         put_text('\nLet\'s get started! ')
#         select_recommendation_system()

# def fun():
#     put_markdown('## Please wait! Your request is being processed!')
#     pywebio.session.hold()
#     description = dataset2['description']
#     count =0
#     put_processbar('bar')
#     for i in range(1, 11):
#         set_processbar('bar', i / 10)
#         time.sleep(0.1)
#     for place in cities:
#         put_html('<hr>')
#         put_markdown("# *`%s`*" % place)
#         pic = 'Images/' + str(place) + '.jpg'
#         img = open(pic, 'rb').read()
#         put_image(img, width='1500px')
#         desc = description[count]
#         desc = desc.strip()
#         desc = desc.replace('-','')
#         put_text(desc)
#         count+=1
#         #put_image(img) to get the original size
#     #style(put_text('In case of copyright issues, please drop an email to rishabh20118@iiitd.ac.in'), 'color:red')
#     put_markdown("# *In case of copyright issues, please drop an email to `rishabh20118@iiitd.ac.in`*")
#     img = open('Images/India_1.jpg', 'rb').read()
#     put_image(img, width='1500px')
    
# def select_recommendation_system():
#     recommendation_system = select('Which type of recommendation system would you prefer?', ['BM25 based Recommendation System', 'TF-IDF(Content based)','Count-Vectorizer(Content Based)','Word2Vec(Content Based)'])
#     #BM25 based Recommendation System
#     if(recommendation_system == 'BM25 based Recommendation System'):
#         put_text('BM25 based Recommendation System is a free text based recommendation system.')
#         free_text = textarea('Enter a free text', rows = 3, placeholder = 'Write anything...\n\'snow winter nature trekking\' ... \'lake boating waterfall tiger\' ... \'market clothes nights history\' ... \'beach cruise camping boats ships\' ... \'temples hills altitude winter line\' ... \'lion safari forests camping nature\' ... ')
#         query = free_text
#         corpus = document_list
#         tokenized_corpus = [doc.split(" ") for doc in corpus]
#         bm25 = BM25Okapi(tokenized_corpus)
#         query=query.lower()
#         query=lemmatizer.lemmatize(query)
#         put_text('Query after preprocessing '+ query)
#         tokenized_query = query.split(" ")
#         doc_scores = bm25.get_scores(tokenized_query)
#         put_text(query)
#         display_recommendations(doc_scores)
        
        
#     if(recommendation_system == 'TF-IDF(Content based)'):
#         put_text('Content Based recommendation system.')
#         free_text = textarea('Enter a free text', rows = 3, placeholder = 'Write anything...\n\'snow winter nature trekking\' ... \'lake boating waterfall tiger\' ... \'market clothes nights history\' ... \'beach cruise camping boats ships\' ... \'temples hills altitude winter line\' ... \'lion safari forests camping nature\' ... ')
#         query = free_text
#         corpus = document_list
#         tokenized_corpus = [doc.split(" ") for doc in corpus]
#         bm25 = BM25Okapi(tokenized_corpus)
#         query=query.lower()
#         query=lemmatizer.lemmatize(query)
#         put_text('Query after preprocessing '+ query)
#         tokenized_query = query.split(" ")
#         doc_scores = tf_idf_sim_score(query,document_list)
#         put_text(query)
#         display_recommendations(doc_scores)
        
#     if(recommendation_system == 'Count-Vectorizer(Content Based)'):
#         put_text('Content Based recommendation system.')
#         free_text = textarea('Enter a free text', rows = 3, placeholder = 'Write anything...\n\'snow winter nature trekking\' ... \'lake boating waterfall tiger\' ... \'market clothes nights history\' ... \'beach cruise camping boats ships\' ... \'temples hills altitude winter line\' ... \'lion safari forests camping nature\' ... ')
#         query = free_text
#         corpus = document_list
#         tokenized_corpus = [doc.split(" ") for doc in corpus]
#         bm25 = BM25Okapi(tokenized_corpus)
#         query=query.lower()
#         query=lemmatizer.lemmatize(query)
#         put_text('Query after preprocessing '+ query)
#         tokenized_query = query.split(" ")
#         doc_scores = countvec_sim_score(query,document_list)
#         put_text(query)
#         display_recommendations(doc_scores)  
        
#     if(recommendation_system == 'Word2Vec(Content Based)'):
#         put_text('Content Based recommendation system.')
#         free_text = textarea('Enter a free text', rows = 3, placeholder = 'Write anything...\n\'snow winter nature trekking\' ... \'lake boating waterfall tiger\' ... \'market clothes nights history\' ... \'beach cruise camping boats ships\' ... \'temples hills altitude winter line\' ... \'lion safari forests camping nature\' ... ')
#         query = free_text
#         corpus = document_list
#         tokenized_corpus = [doc.split(" ") for doc in corpus]
#         bm25 = BM25Okapi(tokenized_corpus)
#         query=query.lower()
#         query=lemmatizer.lemmatize(query)
#         put_text('Query after preprocessing '+ query)
#         tokenized_query = query.split(" ")
#         try:
#             doc_scores = word2vec_sim_score(query,document_list)
#         except Exception as ex:
#             put_text(ex)
#         put_text(query)
#         try:
#             display_recommendations(doc_scores)   
#         except Exception as ex:
#             put_text(ex)
            
        
# def display_recommendations(document_scores):
#     clear();
#     global main_document_scores
#     try:
#         main_document_scores = document_scores.tolist()
#         put_text(''+str(len(main_document_scores)))
#     except Exception as ex:
#         put_text(ex)
   
#     try:
#         clear('BTV')
#     except:
#         pass
#     img = open('Images/DesiSafar Logo.jpg', 'rb').read()
#     put_image(img, width='900px')
#     put_markdown('# **IR Project - Group Number 4**')
#     recommendations = int(input('Enter number of recommendations you want : '))
#     ds= document_scores
#     recommendations_index =ds.argsort()[-recommendations:][::-1]
#     print(recommendations_index)
#     cities = dataset2['City']
#     desc = dataset2['description']
    
#     displayed_recommendations = []
#     displayed_recommendations_index ={}
#     print()
#     for i in range(recommendations_index.shape[0]):
#       put_html('<hr>')
#       put_html('<hr>')
#       pic = 'Images/' + str(cities[recommendations_index[i]]) + '.jpg'
#       img = open(pic, 'rb').read()
#       put_image(img, width='1500px')
#       put_markdown("# *`%s`*" % cities[recommendations_index[i]])
#       t=desc[recommendations_index[i]]
#       t = t.strip()
#       t = t.replace('-','')
#       put_text(t)
#       displayed_recommendations.append(cities[recommendations_index[i]])
#       displayed_recommendations_index[cities[recommendations_index[i]]]=recommendations_index[i]
#       print('-------------------------------------------------------')
#     try:
#         selected_recommendation = select('Explore :', displayed_recommendations)
#         display_details(displayed_recommendations_index[selected_recommendation])
#     except Exception as ex:
#         put_text(ex)
    
# def display_recommendations_temp():
#     put_text('Displayin more recommendations.....'+str(len(main_document_scores)))
#     try:
#         doc_score = np.array(main_document_scores)
#         display_recommendations(doc_score)
#     except Exception as ex:
#         put_text(ex)
#     put_text(''+str(doc_score.shape[0]))
# def display_details(selected_index):
#     row = dataset2.iloc[selected_index]
#     clear()
#     put_markdown('# **IR Project - Group Number 4**')
#     put_markdown("# *`%s`*" % cities[selected_index])
#     put_html('<hr>')
#     pic = 'Images/' + str(cities[selected_index]) + '.jpg'
#     img = open(pic, 'rb').read()
#     put_image(img, width='1500px')
    
#     put_markdown(r""" #  Description""", lstrip=True)
#     t = row['description']
#     t = t.strip()
#     t = t.replace('-','')
#     put_text(t)
#     try:
        
# #         put_buttons([('Best time to visit',str(selected_index)), ('How to reach',str(selected_index))], onclick=[btv, htr])
# #         put_buttons([dict(label='Best time to visit', value=str(selected_index), color='primary')],onclick = btv)
# #         put_buttons([dict(label='How to reach', value=str(selected_index), color='primary')],onclick = htr)
# #         put_buttons([dict(label='Places to visit', value=str(selected_index), color='primary')],onclick = ptv)
#         put_grid([
#                 [put_buttons([dict(label='Best time to visit', value=str(selected_index), color='primary')],onclick = btv),
#                  put_buttons([dict(label='How to reach', value=str(selected_index), color='primary')],onclick = htr),
#                put_buttons([dict(label='Places to visit', value=str(selected_index), color='primary')],onclick = ptv),
#                  put_buttons(['Back'], onclick=[display_recommendations_temp]),
#                 put_buttons(['Home'], onclick=[choices])],
                
               
#             ], cell_width='150px', cell_height='100px')
#         pywebio.session.hold()
        
#     except Exception as ex:
#         put_text(ex)
    

    
# def btv(selected_index):
#     selected_index = int(selected_index)
#     try:
#         set_scope('BTV',-1,-1,'clear')
#         clear('BTV')
#         row = dataset2.iloc[selected_index]
#         put_markdown(r""" #  Best time to visit""", lstrip=True,scope='BTV')
#         t = row['best to time visit']
#         t = t.strip()
#         t = t.replace('-','')
#         put_text(t,scope='BTV')
#     except Exception as ex:
#         put_text(ex)
        
# def ptv(selected_index):
#     selected_index = int(selected_index)
#     try:
#         set_scope('BTV',-1,-1,'clear')
#         clear('BTV')
#         row = dataset2.iloc[selected_index]
#         put_markdown(r""" #  Near by places to visit""", lstrip=True,scope='BTV')
#         t = row['sites']
#         t = t.strip()
#         t = t.replace('-','')
#         put_text(t,scope='BTV')
#     except Exception as ex:
#         put_text(ex)
    
# def htr(selected_index):
#     selected_index = int(selected_index)
#     try:
#         row = dataset2.iloc[selected_index]
#         set_scope('BTV',-1,-1,'clear')
#         clear('BTV')
#         put_markdown(r""" #  How to reach""", lstrip=True,scope='BTV')
#         t = row['how to reach']
#         t = t.strip()
#         t = t.replace('-','')
#         put_text(t,scope='BTV')
#     except Exception as ex:
#         put_text(ex)
    


# In[ ]:


# app = Flask(__name__)
# app.add_url_rule('/', 'webio_view', webio_view(choices), methods=['GET', 'POST', 'OPTIONS'])
# app.run()


# In[ ]:





# In[ ]:




