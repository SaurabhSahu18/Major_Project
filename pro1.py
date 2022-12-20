from random import choices
import streamlit as st
import pandas as pd
import plotly_express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from validators import url
from project1 import *
import analysis as aly
from PIL import Image
import os
plt.style.use('ggplot')
graphchoice=['Scrapped_content','Sentiment_analysis']
choice=st.sidebar.selectbox("Choice",graphchoice)
if choice == graphchoice[0]:
    st.title("Sentiment Analysis on Product review using Web Scraping.")

    st.subheader("This model will display the exact analysis on product by using the customer's reviews.")
    st.subheader("The way to proceed with this model:")
    image= Image.open('procedure.jpg')
    st.image(image,caption='1.Pattern to copy url of any product:')
    product_url = None

    with st.form('form',clear_on_submit=True):
        product_url = st.text_input("Enter product URL here:")
        product_title = st.text_input("Enter a Product name:")
        submit =st.form_submit_button('Submit the product url:')
    if submit and product_url and product_title:
        pid = product_url.split('/')[5]
        driver, link = start(product_url)
        page = 1
        review_list = []
        while True:
            try:
                data = extract(driver,link,page)
                if len(data)>0:
                    review_list.extend(data)
                    page+=1
                else:
                    # st.write('please provide product url')
                    print("no data")
                    break
            except Exception as e:
                st.write(e)
                break
            
        driver.close()
        if len(review_list)>0:
            save(review_list,f'data/product_{pid}_reviews.csv',title=product_title)
            # st.write(review_list)
            st.success('Data scrapped successfully')
        else:
            st.write("no reviews")
            st.error("Reviews doesn't exist")


elif choice == graphchoice[1]:
    if not os.path.exists('data'):
        os.mkdir('data')
    filelist = []
    for idx,file in enumerate(os.listdir('data')):
        filelist.append({
            'name': f'Product Reviews {idx+1}',
            'path': f'data/{file}'
        })
    paths = [p['path']  for p in filelist]
    st.title("Select a File for Analysis")
    sel_file = st.selectbox('Select a file for analysis',options=paths)
    if sel_file:
        st.title("Raw Data")
        st.write('Selected file',sel_file)
        product = aly.get_df(sel_file)
        product['sentence_wise_sentiment']=product.content.apply(aly.get_sentiment_data)
        product['review_sentiment']=product.content.apply(aly.get_review_sentiment)
        product['review_subjectivity']=product.content.apply(aly.get_review_subjectivity)
        product['sentence_count']=product.content.apply(aly.get_review_sentence_count)
        product['sentiment']=product.review_sentiment.apply(aly.get_sentiment)
        if st.sidebar.checkbox("show raw data"):
            st.write(product)

        st.title("Data Visualization")
        st.header('Pie Chart:Analyze product on your fashion')
        pie_out = product.sentiment.value_counts().reset_index()
        fig1 = px.pie(pie_out,names='index',values='sentiment',width=900,height=500) 
        st.plotly_chart(fig1)
        
        st.header('BarChart: Sentiment based analysis')
        background_color = 'lightgrey'
        viz = product.sentiment.value_counts().reset_index()
        fig2 = px.bar(viz, viz['index'],viz['sentiment'],width=900,height=500)
        fig2.update_layout( plot_bgcolor=background_color)
        st.plotly_chart(fig2)
        
        st.header('BoxChart:Analysis based on rating and review sentiment')
        fig22 = px.box(product, 'rating','review_sentiment',width=900,height=500)
        fig22.update_layout( plot_bgcolor=background_color)
        st.plotly_chart(fig22)
        
        st.header('ScatterPlotChart: Analysis based on review sentiment and review subjectivity')
        fig3 = px.scatter(product, x ='review_sentiment',y='review_subjectivity',width=900,height=500)
        fig3.update_layout(
        plot_bgcolor=background_color)
        st.plotly_chart(fig3)
        
        st.header('Histogram:Analysis based on review sentiment and review subjectivity')
        background_color = 'lightgrey'
        fig4 = px.histogram(product, x ='review_sentiment',y='review_subjectivity',width=900,height=500)
        fig4.update_layout(
        plot_bgcolor=background_color)
        st.plotly_chart(fig4)

        
else:
    st.write('')





