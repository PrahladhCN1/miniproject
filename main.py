import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

st.title("Constitution Article Summarizer")

# Simplified search and summarization functions
def search_article(keyword=None):
    url = 'https://prsindia.org/billtrack/category/all'
    html = requests.get(url)
    s = BeautifulSoup(html.content, 'html.parser')
    articles = []
    key = s.find_all('div', class_="views-row")
    for i in key:
        if keyword.lower() in i.text.lower():
            article_number = i.find('a').text.strip()
            link = 'https://prsindia.org' + i.find('a').get('href')
            desc = i.find('div', class_='content').text
            articles.append({"article_id": article_number, "article_desc": desc, "link": link})
    return articles

summarizer = pipeline("summarization", model="google/pegasus-xsum")

def summarize_text(text, max_length=120):
    return summarizer(text, max_length=max_length, min_length=20, do_sample=False)[0]['summary_text']

keyword = st.text_input("Enter the keyword:")
if st.button("Search"):
    articles = search_article(keyword=keyword)
    if not articles:
        st.error("No articles found.")
    else:
        for article in articles:
            st.write(f"**{article['article_id']}**: {article['article_desc'][:100]}...")

article_text = st.text_area("Enter article text to summarize:")
summary_length = st.slider("Select summary length", 50, 500, 120)
if st.button("Summarize"):
    if article_text:
        summary = summarize_text(article_text, max_length=summary_length)
        st.write("**Summary**")
        st.write(summary)
    else:
        st.error("Please enter text to summarize.")
