import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, PegasusForConditionalGeneration, PegasusTokenizer
import requests
from bs4 import BeautifulSoup
import unicodedata
from googletrans import Translator
import sentencepiece

# Load the sentiment analysis model
@st.cache_resource
def load_sentiment_model():
    tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    return tokenizer, model

tokenizer, sentiment_model = load_sentiment_model()

# Load the summarization model
@st.cache_resource
def load_summarization_model():
    model_name = "google/pegasus-cnn_dailymail"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

summarization_tokenizer, summarization_model = load_summarization_model()

# Load translation model
@st.cache_resource
def load_translator():
    return Translator()

translator = load_translator()

def get_bill_info(url):
    html = requests.get(url)
    s = BeautifulSoup(html.content, 'html.parser')
    key = s.find_all('a', title='PRS Bill Summary')
    for i in key:
        k = i.get('href')
    url2 = 'https://prsindia.org' + k
    html2 = requests.get(url2)
    s1 = BeautifulSoup(html2.content, 'html.parser')
    key1 = s1.find(class_='body_content')
    return key1

def extract_key_points(soup):
    key_points = []
    k = soup.find_all('ul')
    if not k:
        k = soup.find_all('ul')
        for i in range(3, len(k) - 5):
            key_points.append(unicodedata.normalize("NFKD", k[i].get_text(strip=True)))
    elif len(k) == 1:
        k = soup.find_all('span', style='font-size:16px')
        for i in k:
            key_points.append(unicodedata.normalize("NFKD", i.get_text(strip=True)))
    else:
        for i in k:
            key_points.append(unicodedata.normalize("NFKD", i.get_text(strip=True)))
    return key_points

def summarize_text(text):
    inputs = summarization_tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = summarization_model.generate(inputs["input_ids"], max_length=300, min_length=150, num_beams=4, early_stopping=True)
    summary = summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def translate_text(text, dest_language):
    translation = translator.translate(text, dest=dest_language)
    return translation.text

st.title('Bill Analysis and Summarization')

url = st.text_input('Enter the URL of the bill', 'https://prsindia.org/billtrack/the-jammu-and-kashmir-local-bodies-laws-amendment-bill-2024')
if st.button('Analyze'):
    with st.spinner('Fetching and analyzing the bill...'):
        bill_content = get_bill_info(url)
        key_points = extract_key_points(bill_content)
        full_text = " ".join(key_points)
        summary = summarize_text(full_text)
        
        st.subheader('Bill Summary')
        st.write(summary)
        
        st.subheader('Sentiment Analysis')
        tokens = tokenizer.encode(summary, return_tensors='pt')
        result = sentiment_model(tokens)
        sentiment = ['Very Opposing', 'Slightly Negative', 'Needs Improvement', 'Slightly Positive', 'Complete Grant']
        sentiment_result = sentiment[int(torch.argmax(result.logits))]
        st.write(sentiment_result)
        
        st.subheader('Translations')
        translated_text_hi = translate_text(summary, 'hi')
        translated_text_kn = translate_text(summary, 'kn')
        translated_text_te = translate_text(summary, 'te')
        
        st.write("Translated to Hindi:", translated_text_hi)
        st.write("Translated to Kannada:", translated_text_kn)
        st.write("Translated to Telugu:", translated_text_te)
