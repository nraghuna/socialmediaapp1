import pandas as pd
from flask import Flask, jsonify, request
import requests
import pandas as pd
from flask_cors import CORS
import datetime
#from datetime import datetime
import json
import sqlite3
import csv
app = Flask(__name__)
import numpy as np
CORS(app)
import ast

from bs4 import BeautifulSoup
import requests
import unicodedata
import re
from collections import Counter
import underthesea
from underthesea import word_tokenize
from urllib.parse import urlparse

import math
import sys
import shelve
import pickle
import os


def clean_text(text):
    pattern = re.compile(r'[^áàảãạâấầẩẫậăẵẳắằặđéèẻẽẹêếềểễệíìịỉĩóòõỏọôốồổộỗơớờởỡợúùũủụưứừửữựýỳỷỹỵ\sa-z_]')
    return re.sub(pattern, ' ', text)

def remove_stopwords(text,stopwords_set):
    tokens = [token for token in text.split() if token not in stopwords_set]
    return tokens

def preprocess_text(text,stopwords):
    processed_text = word_tokenize(text, format='text')
    processed_text = processed_text.lower()
    processed_text = clean_text(processed_text)
    tokens = remove_stopwords(processed_text, stopwords)
    return tokens


def tf(freq):
    return 1 + math.log(freq)


def idf(df, num_docs):
    return math.log(num_docs / df)

def computerank(pages,visited):
        d = 0.8
        numloops = 10
        ranks = {}
        npages = len(visited)
        for url in visited:
                ranks[url] = 1.0 / npages
        for i in range(numloops):
            newranks = {}
            for url in visited:
                newrank = (1 - d) / npages
                for page in pages:
                    for key,value in page.items():
                        if url in page[key]:
                            newrank = newrank + d * ranks[url] / len(page)
                newranks[url] = newrank
            ranks = newranks
        return ranks


def computeranks(pages, visited,keyword):
    d = 0.8
    numloops = 10
    ranks = {}
    npages = len(visited)
    for url in visited:
        ranks[url] = 1.0 / npages
    for i in range(numloops):
        newranks = {}
        for url in visited:
            newrank = (1 - d) / npages
            for page in pages:
                for key, value in page.items():
                    if url in page[key] and keyword in value:
                        newrank = newrank + d * ranks[url] / len(page)
            newranks[url] = newrank
        ranks = newranks
    return ranks


def makeindex(text,url):
    page={}
    for word in text:
        if word not in page:
            page[word]=[]

    for word in text:
        if url not in page[word]:
            page[word].append(url)

    return page

def extract_urls_from_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    urls = set()

    for anchor in soup.find_all('a'):
        href = anchor.get('href')
        if href and href.startswith('http'):
            urls.add(href)

    base_url = urlparse(url).scheme + '://' + urlparse(url).netloc
    urls = [base_url + u if u.startswith('/') else u for u in urls]

    return urls

def indexlookup(page,keyword):
    pages=[]
    for key, value in page.items():
        if keyword in value[0]:
            pages.append(page[key])
    return pages

def crawl(keyword):
    visited_urls=[]

    #map_url= 'http://en.wikipedia.org/sitemap.xml'
    #maps= urlextractor(map_url)
    maps= extract_urls_from_page('https://example.com')
    print(maps)
    pages=[]
    index=[]
    for url in maps:
        #urls= urlextractor(m)
        #for url in urls[:50]:
            #if url not in visited_urls:
        visited_urls.append(url)
        text = textextractionfromurl(url)
        #print(text)
        page= makeindex(url,text)
        #print(page)
        index= indexlookup(page,keyword)
        #print(index)
        pages.append(page)
                ##yield url, text
    ranks= computerank(pages,visited_urls)
    newranks=computeranks(pages,visited_urls,keyword)
    return index,newranks



def urlextractor(url):
    res=requests.get(url)
    if res.status_code!=200:
        return []

    soup= BeautifulSoup(res.text,'html.parser')
    urls=[]
    url_tags=soup.find_all('url')
    for url in url_tags:
        loc= url.find('loc')
        urls.append(loc.text)
    return urls

def textextractionfromurl(url):
    res = requests.get(url)
    if res.status_code != 200:
        return ''
    soup = BeautifulSoup(res.text, 'html.parser')
    text_elements = soup.find_all(['h1', 'h2', 'p'])
    chunks = [element.text.strip() for element in text_elements if element.text.strip()]
    return '\n'.join(chunks)


def textextractionfromurls(url):
    res= requests.get(url)
    if res.status_code!=200:
        return ''
    soup=BeautifulSoup(res.text,'html.parser')
    r1=soup.find('h1',class_='title_news_detail mb10')
    r2= soup.find('h2',class_='title_news_detail mb10')
    if (r1 is None) or (r2 is None):
        return ''
    chunks=[]
    chunks.append(r1.text.strip())
    chunks.append(r2.text.strip())
    content= soup.find_all('p',class_='Normal')
    for q in content[:2]:
        if q.find('a') is None:
            chunks.append(q.get_text().strip())
    return '\n'.join(chunks)


@app.route('/search', methods=['POST','GET'])
def searches():
    if request.method == 'POST':
            data = request.get_json()
            input= data.get('userInput')
            #print(input)
            df = pd.DataFrame.from_dict(data, orient='index')
            df = df.transpose()
            df.to_csv('search.csv', index=False, header=True)
            response= json.dumps(data)
            return response
    elif request.method == 'GET':
        df2= pd.read_csv('search.csv')
        json_data1 = df2.to_json(orient='records')
        datass = json.loads(json_data1)
        print(datass[0]['userInput'])
        index,newrank=crawl(datass[0]['userInput'])
        #print(newrank)
        return jsonify(newrank)

if __name__ == '__main__':
    #h=kmeans()
    #print(h)
    app.run(debug=True)
    ###index,newrank=crawl("We act as both the registrant")
    #index,newrank=crawl("Example domains")
    #print(newrank)
    #app.run(debug=True)