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
from nltk.corpus import stopwords
# import tensorflow.keras.layers as layers
# import tensorflow.keras as keras
# keras.backend.clear_session()
# encoding_dim = 53255
from surprise import Dataset
from surprise import Reader
import pandas as pd 
import pandas as pd 
from surprise import SVD
from surprise import SVDpp
from surprise import Dataset
from surprise.model_selection import cross_validate
from scipy import spatial
from surprise import KNNWithMeans



dataset2 = pd.read_csv('wiki_voyage_correct.csv')
dataset1 = pd.read_csv('final_dataset_wo_duplicates.csv')
main_document_scores=[]
total_time = 0


# def get_recommendations_score(user_id):
data2 = pd.read_csv('ratings_data.csv')
data = data2
poptable = pd.read_csv('poptable.csv')
poptable['State.Name'] = pd.Series(poptable['State.Name'], dtype="string")
data3  = pd.read_csv('final_dataset_wo_duplicates.csv')
#createa a dictionary mapping places to number id
city_to_num = dict()
num_to_city = dict()
k=1
cities_rated=0
print('Imports done')
for i in range(3,len(data.columns)):
    city_to_num[data.columns[i]] = k 
    num_to_city[k] = data.columns[i]
    k+=1
#create userid cityid rating
print('Imports and mapping done')
# res = []
# for i in range(len(data.index)):
#     for j in range(3,len(data.columns)):
#         l=[]
#         if data.iloc[i][j] != 0:
#             l.append(i) #append user id
#             l.append(city_to_num[data.columns[j]]) #append city id
#             l.append(data.iloc[i][j]) # append user rating for that city
#             res.append(l)

# temp = np.asarray(res)
# ratings = pd.DataFrame(temp,columns=['userid','cityid','ratings'])

with open('ratings.pkl', 'rb') as file:
      
    # Call load method to deserialze
    ratings = pickle.load(file)

print('DataFrame created')
# reader = Reader(rating_scale=(1, 5))
# data = Dataset.load_from_df(ratings, reader)
# sim_options = {
# "name": "msd",
# "user_based": True,  # Compute  similarities between items,
# "min_support" : 3
# }
# with open('data.pkl', 'rb') as file:
      
#     # Call load method to deserialze
#     data = pickle.load(file)
    
# with open('reader.pkl', 'rb') as file:
      
#     # Call load method to deserialze
#     reader = pickle.load(file)

# with open('sim_options.pkl', 'rb') as file:
      
#     # Call load method to deserialze
#     sim_options = pickle.load(file)
# algo = KNNWithMeans(sim_options=sim_options,measures=['RMSE', 'MAE'],k=3,min_k=0)
# trainingSet = data.build_full_trainset()
# algo2 = SVDpp()
# cross_validate(algo2, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
# print('CV 1 Done')
# cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)
# print('CV 2 Done')

try:
    with open('algo.pkl', 'rb') as file:
        print('Loading model.....')
        # Call load method to deserialze
        algo = pickle.load(file)
except:
    print('model loading failed, creating a new model')
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings, reader)
    sim_options = {
    "name": "msd",
    "user_based": True,  # Compute  similarities between items,
    "min_support" : 3
    }
    algo = KNNWithMeans(sim_options=sim_options,measures=['RMSE', 'MAE'],k=3,min_k=0)
    trainingSet = data.build_full_trainset()
    algo.fit(trainingSet)
    with open('algo.pkl', 'wb') as file:
        pickle.dump(algo, file)
    
    
# with open('algo2.pkl', 'rb') as file:
      
#     # Call load method to deserialze
#     algo2 = pickle.load(file)
    
# with open('trainingSet.pkl', 'rb') as file:
      
#     # Call load method to deserialze
#     trainingSet = pickle.load(file)


def get_pred(userid,threshold,algo):
#   state_of_user = data2[data2[' ID']==userid]['Which state do you belong to ']
#   state_of_user = state_of_user.to_list()[0].lower()
  conn = sqlite3.connect('test2.db')
  cursor = conn.cursor()
  cursor = conn.execute('Select State from user_details where user_id=?', (userid,))
  state_of_user=''
  for row in cursor:
    state_of_user=row[0]
  state_of_user = state_of_user.lower()
  lat_user = poptable[poptable['State.Name'].str.match(state_of_user)]['latitude'].to_list()[0]
  long_user = poptable[poptable['State.Name'].str.match(state_of_user)]['longitude'].to_list()[0]
  # long_user = poptable[poptable['State.Name']==state_of_user]['longitude']
  user_loc = [lat_user,long_user]
  print("state of user",state_of_user)
  print(lat_user,long_user)
  result = []
  temp_res = []
  col_names = [i for i in range(1,287)]
  temp = pd.DataFrame(columns=col_names)
  for k,v in city_to_num.items():
      pred = algo.predict(userid,v)
      # print(pred)
      city = num_to_city[v]
      city = city.replace('Rate [','')
      city = city.replace(']','')
      # print(city)
      city_state = data3[data3.City==city]['State']
      city_state  = city_state.to_list()[0].lower()
      # print(city_state)
      lat_city = poptable[poptable['State.Name'].str.match(city_state)]['latitude'].to_list()[0]
      long_city = poptable[poptable['State.Name'].str.match(city_state)]['longitude'].to_list()[0]
      # print(lat_city,long_city)
      print(pred)
      city_loc = [lat_city,long_city]
      cos_sim = 1 - spatial.distance.cosine(user_loc, city_loc)
      est = pred.est/5
      total_score = .8*cos_sim + .2*est
      if total_score >= threshold:
        temp_res.append(total_score)
        result.append([k,total_score])
      else:
        temp_res.append(0)

  temp_res = np.array(temp_res)
  print(temp_res.shape)
  temp=pd.DataFrame(temp_res,index=col_names)
  conn.close()
  return temp,result
      





def word2vec_sim_score(q,doc_list):
  global document_list
  if len(doc_list) ==287:
    doc_list = doc_list[:-1]
  doc_list.append(q)
  stemmed_doc_list, doc_vector = list(), list()
  for doc in doc_list:
    stemmed_doc_list.append(doc.split(" "))
  # print(stemmed_doc_list)
  word2vec = Word2Vec(stemmed_doc_list,min_count=1,vector_size=1000)
  for index,row in enumerate(stemmed_doc_list):
    model_vector = (np.mean([word2vec.wv[token] for token in row], axis=0)).tolist()
    doc_vector.append(model_vector)
  sim = cosine_similarity(doc_vector)
  doc_list = doc_list[:-1]
  document_list = doc_list 
  return np.asarray(sim[-1][:-1])

def tf_idf_sim_score(q,doc_list):
  global document_list
  tfvec = TfidfVectorizer(stop_words='english')
  if len(doc_list) ==287:
    doc_list = doc_list[:-1]
  doc_list.append(q)
  vec_corpus = tfvec.fit_transform(doc_list)
  sim = cosine_similarity(vec_corpus)
  doc_list = doc_list[:-1]
  document_list = doc_list
  return np.asarray(sim[-1][:-1])

def countvec_sim_score(q,doc_list):
  global document_list
  countvec = CountVectorizer(stop_words='english')
  if len(doc_list) ==287:
    doc_list = doc_list[:-1]
  doc_list.append(q)
  vec_corpus = countvec.fit_transform(doc_list)
  sim = cosine_similarity(vec_corpus)
  doc_list = doc_list[:-1]
  document_list = doc_list
  return np.asarray(sim[-1][:-1])

document_list=[]
lemmatizer = WordNetLemmatizer() 
i=0
cities = list(dataset2['City'])


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
#       document=lemmatizer.lemmatize(document)
      tokens = document.split()
      tokens = [lemmatizer.lemmatize(word) for word in tokens if not word in stopwords.words('english')]
      text = ' '.join(tokens)
      document_list.append(text)
      i+=1
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


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(document_list)
vocab = vectorizer.vocabulary_
idf_values = vectorizer.idf_
dense = X.toarray()
x_train_1 = np.float32(dense)
# print('loading keras model...')
# encoder = keras.models.load_model('model2')
# X_encoded = encoder.predict(x_train_1)
# print('model loaded')

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
        put_buttons([dict(label='Explore', value=str(count), color='primary')],onclick=explore_city)
        count+=1
        #put_image(img) to get the original size
    #style(put_text('In case of copyright issues, please drop an email to rishabh20118@iiitd.ac.in'), 'color:red')
#     put_markdown("# *In case of copyright issues, please drop an email to `rishabh20118@iiitd.ac.in`*")
    img = open('Images/India_1.jpg', 'rb').read()
    pywebio.session.hold()
    put_image(img, width='1500px')
   

def explore_city(count):
    selected_index = int(count)
    clear()
    row = dataset2.iloc[selected_index]
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
                 put_buttons([dict(label='Hotel Details', value=str(selected_index), color='primary')],onclick = hotels),
                 
                put_buttons(['Home'], onclick=[choices])],
                
               
            ], cell_width='150px', cell_height='100px')
        pywebio.session.hold()
        
    except Exception as ex:
        put_text(ex)
    


def select_recommendation_system():
    recommendation_system = select('Which type of recommendation system would you prefer?', ['BM25 based Recommendation System', 'TF-IDF(Content based)','Count-Vectorizer(Content Based)','Word2Vec(Content Based)','Collaborative Filtering'])
    global total_time
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
        start_time = time.time()
        doc_scores = bm25.get_scores(tokenized_query)
        time.sleep(0.3)
        put_text(query)
        end_time = time.time()
        total_time = end_time - start_time
        try:
            display_recommendations(doc_scores)
        except Exception as ex:
            put_text(ex)
        
        
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
        start_time = time.time()
        doc_scores = tf_idf_sim_score(query,document_list)
        put_text(query)
        end_time = time.time()
        total_time = end_time - start_time
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
        start_time = time.time()
        doc_scores = countvec_sim_score(query,document_list)
        put_text(query)
        end_time = time.time()
        total_time = end_time - start_time
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
        start_time = time.time()
        try:
            doc_scores = word2vec_sim_score(query,document_list)
            put_text(query)
            end_time = time.time()
            total_time = end_time - start_time
            display_recommendations(doc_scores)
        except Exception as ex:
            put_text(ex)
#         put_text(query)
#         end_time = time.time()
#         total_time = end_time - start_time
#         try:
#             display_recommendations(doc_scores)   
#         except Exception as ex:
#             put_text(ex)
    if(recommendation_system == 'Collaborative Filtering')  :
        account_status = select('Log in or sign up?', ['Log in','Sign Up','Home'])
        if(account_status == 'Log in'):
            log_in_page()
        if(account_status == 'Sign Up'):
            put_text('Sign up selected')
            sign_up_page()
        if(account_status== 'Home'):
            choices()
            
    if(recommendation_system=='Efficient Query Processing') :
        query=''
        try:
            put_text('Fast Query Processing.')
            free_text = textarea('Enter a free text', rows = 3, placeholder = 'Write anything...\n\'snow winter nature trekking\' ... \'lake boating waterfall tiger\' ... \'market clothes nights history\' ... \'beach cruise camping boats ships\' ... \'temples hills altitude winter line\' ... \'lion safari forests camping nature\' ... ')
            query=free_text
            query=query.lower()
            query = query.translate(translator)
        except Exception as ex:
            put_text('Exception in query pre processing :')
        tokens = query.split()
        tokens = [lemmatizer.lemmatize(word) for word in tokens if not word in stopwords.words('english')]
        text = ' '.join(tokens)
        query_vector = np.zeros((X.shape[1]))
        for t in tokens:
          try:
            # print(t)
            t_index = vocab[t]
            query_vector[t_index]= query_vector[t_index] + idf_values[t_index]
            print(t)
          except Exception as ex:
            put_text(ex)
        try:
            query_vec = query_vector.reshape(1,X.shape[1])
            query_encoded = encoder.predict(query_vec)
            query_encoded = query_encoded.reshape(100,)     
            scores_array=np.zeros((len(document_list),))
            dense = X.toarray()
    #         print('Using raw count')
            start_time = time.time()
            for i in range(len(document_list)):
              # print(i)
              doc_vector = X_encoded[i,:]
              result = 1 - spatial.distance.cosine(doc_vector, query_encoded)
              scores_array[i]=result
            end_time = time.time()
            total_time = end_time - start_time
        except Exception as ex:
            put_text(ex)
        try:
            display_recommendations(scores_array)   
        except Exception as ex:
            put_text(ex)
        
            
            
def check_form(data):  # input group validation: return (input name, error msg) when validation fail
        if len(data['name']) > 20:
            return ('name', 'Name too long!')
        if data['age'] <= 0:
            return ('age', 'Age can not be negative!')     
        if data['age'] >= 130:
            return ('age','Enter valid age')
        
def check_log_in(data):
    put_text('Checking log in ')
    conn = sqlite3.connect('test2.db')
    user_id = data['user_id']
    pin = data['password']
    cursor = conn.execute('Select Pin from user_details where user_id=?', (user_id,))
    found=0
    for row in cursor:
      if (row[0] == pin):
        put_text('Successful LOGIN')
        found = 1
        try:
            put_text('Success')
            try:
                conn.close()
                return
            except Exception as ex:
                put_text(ex)
#             t,result= get_pred(user_id,0.9,algo2)
            
#             r = list(t[0])
#             r= np.array(r)
#             put_text(r)
            
#             display_recommendations(r)
            
            
        except Exception as ex:
            put_text(ex)
      else:
        put_text('Wrong PIN or account number')
        found = 0
        conn.close()
        return('user_id','Invalid Details')
    
    if(found==0):
        conn.close()
        return('user_id','Invalid Details')
    
    
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
    put_text('Log in successful')
    try:
        t,result= get_pred(info['user_id'],0.1,algo)

        r = list(t[0])
        r= np.array(r)
        put_text(r)

        display_recommendations(r)
    except Exception as ex:
        put_text(ex)
    

    
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
        select('Select your state',['Andaman And Nicobar Islands',   
'Andhra Pradesh',     
'Arunanchal Pradesh',    
'Assam',    
'Bihar',    
'Chandigarh',    
'Chattisgarh',
'Dadra And Nagar Haveli',     
'Delhi',    
'Goa',    
'Haryana',     
'Himachal Pradesh',    
'Jammu & Kashmir',    
'Jharkhand',    
'Karnataka',     
'Kerala',    
'Lakshadweep Island',    
'Madhya Pradesh',    
'Maharashtra',     
'Manipur' ,   
'Meghalaya',    
'Mizoram'   , 
'Nagaland'   , 
'Odisha'    ,
'Pondicherry',
'Punjab'    ,
'Rajasthan'  ,   
'Sikkim'    
'Tamil Nadu'  ,   
'Tripura'    ,
'Uttar Pradesh',    
'Uttarakhand',
'West Bengal' ,    
'Gujarat',
'Telangana',
'Daman & Diu',
'Ladakh'
]
,name='state',required=True)
],validate=check_form)
    
#     put_markdown("`info = %r`" % info)
    
#     put_text(info['name'],info['age'],info['password'],info['state'])
    insert_details(info)
    pywebio.session.hold()
    
  
def insert_details(info):
    conn = sqlite3.connect('test2.db')
    global cities_rated
    conn.execute('''CREATE TABLE IF NOT EXISTS user_details
             (User_id INTEGER PRIMARY KEY     AUTOINCREMENT,
             Name           TEXT    NOT NULL,
             Age            INT     NOT NULL,
             State        CHAR(50)  NOT NULL,
             Pin CHAR(50));''')
#     conn = sqlite3.connect('test2.db')
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
        cities_rated=0
    except Exception as ex:
        put_text(ex)
        logf.write("Failed to insert {0}:"+ex)
    try:
        for row in cursor:
            put_text('Please note your User ID:  ', row[0])
            account_number=int(row[0])
    except Exception as ex:
        put_text(ex)
    popup('Please note your User ID : '+str(account_number))
    conn.close()
    put_buttons(['Home'], onclick=[choices])
    put_buttons(['Explore'], onclick=[select_recommendation_system])
    try:
        rate_the_cities(account_number)
    except Exception as ex:
        put_text(ex)
    
def rate_the_cities(user_id):
    global cities_rated
    cities_rated=0
    count=0
    for city in cities:
        temp_scope=str(count)
        count+=1
        set_scope(temp_scope,-1,-1,'clear')
        put_html('<hr>',scope=temp_scope)
        put_text('',scope=temp_scope)
        put_text(city,scope=temp_scope)
        try:
            put_grid([
                [put_buttons([dict(label='1', value=str(user_id)+'_'+city+'_'+str(1), color='primary')],onclick = save_ratings),
                 put_buttons([dict(label='2', value=str(user_id)+'_'+city+'_'+str(2), color='primary')],onclick = save_ratings),
               put_buttons([dict(label='3', value=str(user_id)+'_'+city+'_'+str(3), color='primary')],onclick = save_ratings),
                 put_buttons([dict(label='4', value=str(user_id)+'_'+city+'_'+str(4), color='primary')],onclick = save_ratings),
                 put_buttons([dict(label='5', value=str(user_id)+'_'+city+'_'+str(5), color='primary')],onclick = save_ratings),
                 put_buttons([dict(label='Save', value=str(user_id)+'_'+city+'_'+str(6), color='primary')],onclick = save_model)
               
            ]], cell_width='50px', cell_height='20px',scope=temp_scope)
            
        except Exception as ex:
            put_text(ex)
        
    pywebio.session.hold()
   
def save_model(value):
    clear()
    put_text("Saving.... Please wait.. You will be redirected to home page once model is saved",scope='save')
    try:
        global algo
        global cities_rated

        if(cities_rated<5):
            popup('Please rate atleast 5 cities, rated cities :'+str(cities_rated))
            return
        
    except Exception as ex:
        put_text(ex)
        
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings, reader)
    sim_options = {
    "name": "msd",
    "user_based": True,  # Compute  similarities between items,
    "min_support" : 3
    }
    algo = KNNWithMeans(sim_options=sim_options,measures=['RMSE', 'MAE'],k=3,min_k=0)
    trainingSet = data.build_full_trainset()
    algo.fit(trainingSet)
    with open('algo.pkl', 'wb') as file:
        pickle.dump(algo, file)
    with open('ratings.pkl', 'wb') as file:
        pickle.dump(ratings, file)
    clear('save')
    choices()
    
    
    
def save_ratings(value):
    global ratings
    global cities_rated
    try:
        
        put_text()    
#         put_text(value)
        values_list = value.split('_')
        rating = int(values_list[2])
        put_text(rating)
        
#         put_text(values_list)
        user_id = float(values_list[0])
        put_text(user_id)
        city_id = float(cities.index(values_list[1]))
        put_text('city id'+str(city_id))
        temp_scope=str(int(city_id))
        set_scope(temp_scope,-1,-1,'clear')
        clear(temp_scope)
        
#         put_text(user_idcity_id,rating)
        df2 = {'userid': user_id, 'cityid': city_id, 'ratings': rating}
        ratings = ratings.append(df2, ignore_index = True)
        put_text(str(ratings.shape[0]))
        cities_rated+=1
    except Exception as ex:
        put_text(ex)
        
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
#     put_buttons(['Home'], onclick=[choices])
#     put_buttons(['Check another algorithm'], onclick=[select_recommendation_system])
    ds= document_scores
    recommendations_index =ds.argsort()[-recommendations:][::-1]
    print(recommendations_index)
    cities = dataset2['City']
    desc = dataset2['description']
    
    displayed_recommendations = []
    displayed_recommendations_index ={}
    
    put_text('Scores calculated in time : '+str(total_time))
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
        displayed_recommendations.append('Home')
        displayed_recommendations.append('Check another algorithm')
        selected_recommendation = select('Explore :', displayed_recommendations)
        if(selected_recommendation=='Home'):
            choices()
        if(selected_recommendation == 'Check another algorithm'):
            select_recommendation_system()
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
                 put_buttons([dict(label='Hotel Details', value=str(selected_index), color='primary')],onclick = hotels),
                 put_buttons(['Back'], onclick=[display_recommendations_temp]),
                put_buttons(['Home'], onclick=[choices])],
                
               
            ], cell_width='150px', cell_height='100px')
        pywebio.session.hold()
        
    except Exception as ex:
        put_text(ex)
    
def hotels(selected_index):
    selected_index = int(selected_index)
    try:
        set_scope('BTV',-1,-1,'clear')
        clear('BTV')
        row = dataset2.iloc[selected_index]
        put_markdown(r""" #  Hotels Available""", lstrip=True,scope='BTV')
        hotel_names = row['hotel_names']
        hotel_names = list(hotel_names.split("\n"))
        hotel_distance = row['hotel distance']
        hotel_distance = list(hotel_distance.split("\n"))
        hotel_price = row['hotel price']
        hotel_price = hotel_price.replace("&dollar","$")
        hotel_price = list(hotel_price.split("\n"))
        hotel_ratings = row['hotel ratings']
        hotel_ratings = list(hotel_ratings.split("\n"))
        hotels_available = len(hotel_names)
        for i in range(hotels_available):
            put_markdown("# *`%s`*" % hotel_names[i],scope='BTV')
            put_text("Distance from city center : "+hotel_distance[i],scope='BTV')
            put_text("Hotel Price : "+hotel_price[i],scope='BTV')
            put_text("Hotel Rating : "+hotel_ratings[i],scope='BTV')
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
