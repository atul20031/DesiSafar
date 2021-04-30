import numpy as np
import pandas as pd
import os.path
from os import path
from IPython.display import Image
from IPython.display import display
from sklearn.metrics.pairwise import sigmoid_kernel
#!pip install pywebio
from pywebio.input import *
from pywebio.output import *
import time
from pywebio.platform.flask import webio_view
from pywebio import STATIC_PATH
from flask import Flask, send_from_directory
app = Flask(__name__)

df = pd.read_csv('IR New Dataset.csv')
cities = list(df['City'])

#df2 will indicate a dataframe containing destination tags which will be helpful in converting to get similarity matrix
df2 = pd.read_csv('IR New Dataset.csv')
#Reading the same CSV 2 times and modifying one 
df2.drop(['City', 'Tags', 'State', 'Old_age', 'Young_age', 'link 1', 'Avg Expense Per Day', 'Image Link'], axis = 1, inplace = True) 
matrix = df2.values
similarity_matrix = sigmoid_kernel(matrix, matrix)
indices = pd.Series(df.index, index = df['City']).drop_duplicates()

def get_recommendations(title, similarity_matrix = similarity_matrix):
    idx = indices[title]
    similarity_scores_for_a_particular_destination = list(enumerate(similarity_matrix[idx]))
    similarity_scores_for_a_particular_destination = sorted(similarity_scores_for_a_particular_destination, key = lambda x:x[1], reverse = True)
    similarity_scores_for_a_particular_destination = similarity_scores_for_a_particular_destination[1:11]
    recommended_destination_indices = [i[0] for i in similarity_scores_for_a_particular_destination]
    return df['City'].iloc[recommended_destination_indices]

# We will proceed only if user enters a correct place
def check_place(place):
    if(place not in cities):
        return 'Please enter a valid place!'


def content_based_filtering():
    entered_place = input("Enter one of the places from the database which you visited?", type=TEXT, validate=check_place)
    recommendations = get_recommendations(entered_place)
    final_recommendations_list = recommendations.tolist()
    if(entered_destination in final_recommendations_list):
        final_recommendations_list.remove(entered_destination)
    put_html('<hr>')
    put_markdown("Recommendations for you similar to `%s` place are as follows: " % entered_destination)
    put_html('<hr>')
    for place in final_recommendations_list:
        put_html('<hr>')
        put_markdown("# *`%s`*" % place)
        pic = 'Images/' + str(place) + '.jpg'
        img = open(pic, 'rb').read()
        put_image(img, width='1500px')

def select_recommendation_system():
    recommendation_system = select('Which type of recommendation system would you prefer?', ['BM25 based Recommendation System', 'Content-based Filtering','Collaborative Filtering'])
    #BM25 based Recommendation System
    if(recommendation_system == 'BM25 based Recommendation System'):
        put_text('BM25 based Recommendation System is a free text based recommendation system.')
        free_text = textarea('Enter a free text', rows = 3, placeholder = 'Write anything...\n\'snow winter nature trekking\' ... \'lake boating waterfall tiger\' ... \'market clothes nights history\' ... \'beach cruise camping boats ships\' ... \'temples hills altitude winter line\' ... \'lion safari forests camping nature\' ... ')
    
    #Content-based Filtering
    if(recommendation_system == 'Content-based Filtering'):
        put_text('Content-based Filtering is done on the basis of destination type. \nIt will show you travel destination recommendations which are similar to the place you entered.')
        content_based_filtering()
    
    #Collaborative Filtering
    if(recommendation_system == 'Collaborative Filtering'):
        put_text('Collaborative Filtering is done on the basis of user-to-user similarity or item-to-item similarity.')
    
def fun():
    put_markdown('## Please wait! Your request is being processed!')
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
        #put_image(img) to get the original size
    #style(put_text('In case of copyright issues, please drop an email to rishabh20118@iiitd.ac.in'), 'color:red')
    put_markdown("# *In case of copyright issues, please drop an email to `rishabh20118@iiitd.ac.in`*")
    img = open('Images/India_1.jpg', 'rb').read()
    put_image(img, width='1500px')
    #with popup("Subscribe to the page"):
        #put_text("Join other foodies!")
def choices():
    popup('DesiSafar - A Travel Recommendation System', 'Information Retrieval Project [Group Number 4] \n\nRishabh Bafna (MT20118) \nAtul Rawat(MT20___) \nDivisha Bisht (MT20___) \n Aman Dapola (MT20___) \n Vineet Maheshwari (MT20___) \n\n Special thanks to Professor Rajiv Ratn Shah!')
    img = open('Images/DesiSafar Logo.jpg', 'rb').read()
    put_image(img, width='900px')
    put_markdown('# **IR Project - Group Number 4**')
    answer = radio("Choose one", options=['Explore Incredible India!', 'Get Travel Recommendations'])
    if(answer == 'Explore Incredible India!'):
        fun()
    if(answer == 'Get Travel Recommendations'):
        put_text('\nLet\'s get started! ')
        select_recommendation_system()
    
#app.add_url_rule('/rishabh', 'webio_view', webio_view(choices), methods=['GET', 'POST', 'OPTIONS'])
#app.run(host='localhost', port=80)
app.add_url_rule('/', 'webio_view', webio_view(choices), methods=['GET', 'POST', 'OPTIONS'])
app.run()