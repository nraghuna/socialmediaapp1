from flask import Flask, jsonify, request
import requests
import pandas as pd
from flask_cors import CORS
import json
import sqlite3
import csv
app = Flask(__name__)
CORS(app)


data = [{
    "companyName": {
        "chosenName": "Acme Technologies Inc",
        "availability": "Available for registration"
    },
    "businessStructure": {
        "legalStructure": "Limited Liability Company (LLC)",
        "advantages": [
            "Limited liability protection for owners",
            "Flexible management structure",
            "Pass-through taxation"
        ],
        "disadvantages": [
            "Compliance with regulatory requirements",
            "Restrictions on ownership and fundraising"
        ]
    },
    "businessRegistration": {
        "registeredCompanyInfo": {
            "registeredName": "Acme Technologies Inc.",
            "registeredAddress": "123 Main Street, Cityville",
            "ownershipDetails": [
                {"ownerName": "John Doe", "ownershipPercentage": "50%"},
                {"ownerName": "Jane Smith", "ownershipPercentage": "50%"}
            ],
            "registrationFee": "$500"
        }
    },
    "articlesOfIncorporation": {
        "purpose": "Development and sale of innovative software solutions",
        "ownershipStructure": "Members of the LLC with equal ownership shares",
        "governanceProvisions": {
            "meetingFrequency": "Monthly",
            "votingProcedures": "Unanimous consent for major decisions, majority vote for routine matters"
        }
    },
    "shareholdersPartnershipAgreement": {
        "shareholdersPartners": ["John Doe", "Jane Smith"],
        "ownershipDistribution": [
            {"ownerName": "John Doe", "ownershipPercentage": "50%"},
            {"ownerName": "Jane Smith", "ownershipPercentage": "50%"}
        ],
        "decisionMaking": {
            "majorDecisions": "Unanimous consent",
            "routineMatters": "Majority vote"
        }
    },
    "permitsLicenses": {
        "obtainedPermitsLicenses": [
            "Business License",
            "Tax Registrations",
            "Health and Safety Permits"
        ],
        "industrySpecificCertifications": []
    },
    "employmentLaws": {
        "complianceMeasures": [
            "Adherence to equal employment opportunity guidelines",
            "Provision of health insurance and retirement plans",
            "Regular safety inspections and OSHA compliance",
            "Compliance with minimum wage laws and overtime regulations"
        ]
    },
    "intellectualProperty": {
        "trademarkSearch": "No infringement on existing trademarks in the same industry"
    },
    "taxObligations": {
        "taxes": [
            "Income Taxes",
            "Sales Taxes",
            "Payroll Taxes"
        ]
    },
    "contractsAgreements": {
        "clientCustomerAgreements": "Standard terms of service agreements for clients",
        "vendorContracts": "Signed contracts with key vendors",
        "employmentContracts": "Written employment contracts for all employees",
        "partnershipAgreements": "Formalized partnership agreement between owners"
    }
}]

class User:
    def __init__(self, name,age,email,domain):
        self.name= name
        self.age=age
        self.email=email
        self.domain=domain

user=User(None,None,None,None)



@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if request.method == 'POST':
            data = request.get_json()
            name = data.get('name')
            age = data.get('age')
            email = data.get('email')
            j=json.dumps(data)
            h=json.loads(j)
            user.name= data["name"]
            user.age = data["age"]
            user.email = data["email"]
            user.domain = data["domain"]
            user_data = {
            "name": user.name,
            "age": user.age,
            "email": user.email,
            "domain": user.domain
            }
            df = pd.DataFrame.from_dict(data, orient='index')
            df = df.transpose()
            df.to_csv('users.csv', index=False, header=True)
            file = open('users.csv', 'r')
            f = csv.reader(file)
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS my_users (
                                                   column1 TEXT,
                                                   column2 INTEGER,
                                                   column3 TEXT,
                                                   column4 TEXT
                                               )''')
            for row in f:
                cursor.execute("INSERT INTO my_users VALUES (?, ?, ?,?)", row)
            conn.commit()
            conn.close()
            response= json.dumps(user_data)
            return response
    elif request.method == 'GET':
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM my_users")
            rows = cursor.fetchall()
            data = []
            for r in rows:
                if r != rows[0]:
                    rowdata = list(r)
                    data.append(rowdata)
            conn.commit()
            conn.close()
            return jsonify(data)

@app.route('/post', methods=['POST','GET'])
def postsss():
    if request.method == 'POST':
            data = request.get_json()
            posts = data.get('posts')
            posts = {
            "post": posts
            }
            df = pd.DataFrame.from_dict(posts, orient='index')
            df = df.transpose()
            df.to_csv('post.csv', index=False, header=True)
            file = open('post.csv', 'r')
            f = csv.reader(file)
            conn = sqlite3.connect('post.db')
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS my_posts (
                                                   column1 TEXT
                                               )''')
            for row in f:
                cursor.execute("INSERT INTO my_posts VALUES (?)", row)
            conn.commit()
            conn.close()
            response= json.dumps(posts)
            return response
    elif request.method == 'GET':
            conn = sqlite3.connect('post.db')
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM my_posts")
            rows = cursor.fetchall()
            data = []
            for r in rows:
                if r != rows[0]:
                    rowdata = list(r)
                    data.append(rowdata)
            conn.commit()
            conn.close()
            return jsonify(data)

@app.route('/posts', methods=['POST','GET'])
def posts():
    if request.method == 'POST':
        data = request.get_json()
        posts = data.get('posts')
        df = pd.DataFrame({'column1': [posts]})
        df.to_csv('post.csv', index=False, header=True)
        file = open('post.csv', 'r')
        f = csv.reader(file)
        conn = sqlite3.connect('post.db')
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS my_posts (
            column1 TEXT
        )''')
        for row in f:
            cursor.execute("INSERT INTO my_posts VALUES (?)", row)
        conn.commit()
        conn.close()
        response = {'post': posts}
        return jsonify(response)
    elif request.method == 'GET':
        conn = sqlite3.connect('post.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM my_posts")
        rows = cursor.fetchall()
        data = []
        for r in rows:
            if r != rows[0]:
                rowdata = list(r)
                data.append(rowdata)
        conn.commit()
        conn.close()
        return jsonify(data)


@app.route('/comment', methods=['POST','GET'])
def commentsss():
    if request.method == 'POST':
            data = request.get_json()
            comments = data.get('commentss')
            post_id= data.get('post_index')
            commentss = {
            "index": post_id,
            "comment": comments,
            }
            df = pd.DataFrame.from_dict(commentss, orient='index')
            df = df.transpose()
            df.to_csv('comment.csv', index=False, header=True)
            file = open('comment.csv', 'r')
            f = csv.reader(file)
            conn = sqlite3.connect('comment.db')
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS my_comment (
                                                   column1 INTEGER,
                                                   column2 TEXT
                                               )''')
            for row in f:
                cursor.execute("INSERT INTO my_comment VALUES (?, ?)", row)
            conn.commit()
            conn.close()
            response= json.dumps(commentss)
            return response
    elif request.method == 'GET':
            conn = sqlite3.connect('comment.db')
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM my_comment")
            rows = cursor.fetchall()
            data = []
            for r in rows:
                if r != rows[0]:
                    rowdata = list(r)
                    #rowdata = {
                     #   "post_index": r[0],
                      #  "commentss": r[1]
                    #}
                    data.append(rowdata)
            conn.commit()
            conn.close()
            return jsonify(data)

@app.route('/comments', methods=['POST','GET'])
def comments():
    if request.method == 'POST':
        data = request.get_json()
        comments = data.get('commentss')
        post_id = data.get('post_index')
        df = pd.DataFrame({'column1': [post_id], 'column2': [comments]})
        df.to_csv('comment.csv', index=False, header=True)
        file = open('comment.csv', 'r')
        f = csv.reader(file)
        conn = sqlite3.connect('comment.db')
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS my_comment (
            column1 INTEGER,
            column2 TEXT
        )''')
        for row in f:
            cursor.execute("INSERT INTO my_comment VALUES (?, ?)", row)
        conn.commit()
        conn.close()
        response = {'index': post_id, 'comment': comments}
        return jsonify(response)
    elif request.method == 'GET':
        conn = sqlite3.connect('comment.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM my_comment")
        rows = cursor.fetchall()
        data = []
        for r in rows:
            if r != rows[0]:
                rowdata = list(r)
                data.append(rowdata)
        conn.commit()
        conn.close()
        return jsonify(data)


def postscontent():
    if request.method == 'POST':
            data = request.get_json()
            posts = data.get('posts')
            posts = {
            "post": posts
            }
            df = pd.DataFrame.from_dict(posts, orient='index')
            df = df.transpose()
            df.to_csv('post.csv', index=False, header=True)
            file = open('post.csv', 'r')
            f = csv.reader(file)
            conn = sqlite3.connect('post.db')
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS my_posts (
                                                   column1 TEXT,
                                               )''')
            for row in f:
                cursor.execute("INSERT INTO my_posts VALUES (?)", row)
            conn.commit()
            conn.close()
            response= json.dumps(posts)
            return response
    elif request.method == 'GET':
            conn = sqlite3.connect('post.db')
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM my_posts")
            rows = cursor.fetchall()
            data = []
            for r in rows:
                if r != rows[0]:
                    rowdata = list(r)
                    data.append(rowdata)
            conn.commit()
            conn.close()
            return jsonify(data)


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@app.route('/convert', methods=['GET'])
def convert():
        response = requests.get(' http://127.0.0.1:5000/profile')
        data = response.json()
        df=pd.DataFrame(data,columns = ['name','age','email','domain'])
        df.to_csv('convert.csv', index=False, header=False)
        responses = requests.get(' http://127.0.0.1:5000/profile')
        datas = responses.json()
        dfs = pd.DataFrame(datas, columns=['name', 'age', 'email', 'domain'])
        dfs.to_csv('investor.csv', index=False, header=False)

        vectorizer= TfidfVectorizer()
        tfid= vectorizer.fit_transform(df.loc[:,'domain'])
        vec= TfidfVectorizer()
        tff= vec.fit_transform(dfs.loc[:,'domain'])
        cos= cosine_similarity(tfid,tff)
        indexes = []
        ind = df[df.loc[:, 'name'] == 'John'].index[0]
        for i in range(len(cos)):
                indices = []
                for j in cos[i]:
                        if j > 0.3:
                                indices.append(np.where(cos[i] == j)[0][0])
                                indexes.append(indices)
        top_matches = []
        for i in indexes[ind]:
                top_matches.append(dfs.loc[:, 'name'][i])
        return top_matches

def convertToBinaryData(file):
    blobData = file.read()
    return blobData

import base64
import codecs
import io

from flask import send_file, Response,make_response
from io import BytesIO
import webbrowser
import tempfile
import os
from PyPDF2 import PdfFileWriter, PdfReader

from PIL import Image
import io
import zipfile
from flask import Flask, request, send_file, render_template

@app.route('/images', methods=['POST','GET'])
def upload_images():
        if request.method == 'POST':
                conn = sqlite3.connect('images.db')
                cursor = conn.cursor()
                cursor.execute('''CREATE TABLE IF NOT EXISTS my_images (
                                               file BLOB
                                           )''')
                image_data= request.form['images']
                image_bytes=image_data.encode('utf-8')
                encoded_image = base64.b64encode(image_bytes).decode('utf-8')
                cursor.execute("INSERT INTO my_images (file) VALUES (?)", (encoded_image,))
                conn.commit()
                conn.close()
                return 'Image uploaded and stored in database'
        elif request.method == 'GET':
                conn = sqlite3.connect('images.db')
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM my_images")
                rows = cursor.fetchall()
                conn.close()
                images=[]
                for r in rows:
                        base64_string= r[0]
                        base64_string = base64_string.replace("data:application/pdf;base64,", "")
                        decoded_data = base64.b64decode(base64_string)
                        return send_file(
                                io.BytesIO(decoded_data),
                                mimetype='application/pdf',
                        )


import cv2
import base64
import numpy as np
from flask import send_from_directory

from flask import Flask, render_template_string
def encode_image(image_path):
    with open(image_path, 'rb') as f:
        image_data = f.read()
        base64_image = base64.b64encode(image_data).decode('utf-8')
        return base64_image

@app.route('/imagesjpgs', methods=['POST','GET'])
def upload_imagesjpg():
        if request.method == 'POST':
                conn = sqlite3.connect('imagesjpg.db')
                cursor = conn.cursor()
                cursor.execute('''CREATE TABLE IF NOT EXISTS my_imagesjpg (
                                               file BLOB
                                           )''')
                image_data = request.form.getlist('images')

                images = request.form.getlist('images[]')
                for image in images:
                        cursor.execute("INSERT INTO my_imagesjpg (file) VALUES (?)", (image,))

                conn.commit()
                conn.close()
                return 'Image uploaded and stored in database'
        elif request.method == 'GET':
                conn = sqlite3.connect('imagesjpg.db')
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM my_imagesjpg")
                rows = cursor.fetchall()
                conn.close()
                images=[]
                for i,row in enumerate(rows):
                        base64_string= row[0].split(',')[1]
                        image_bytes = base64.b64decode(base64_string)
                        filename = f"image_{i + 1}.png"

                        with open(filename, 'wb') as f:
                                f.write(image_bytes)
                        images.append(filename)
                image_tags = []
                for image in images:
                        base64_image = encode_image(image)
                        image_tag = f'<img src="data:image/png;base64,{base64_image}" alt="Image">'
                        image_tags.append(image_tag)

                html = ' '.join(image_tags)
                return html


@app.route('/imagesjpg', methods=['POST','GET'])
def upload_imagesjpgs():
        if request.method == 'POST':
                conn = sqlite3.connect('imagesjpg.db')
                cursor = conn.cursor()
                cursor.execute('''CREATE TABLE IF NOT EXISTS my_imagesjpg (
                                               file BLOB
                                           )''')
                image_data = request.form.getlist('images')

                images = request.form.getlist('images[]')
                for image in images:
                        cursor.execute("INSERT INTO my_imagesjpg (file) VALUES (?)", (image,))

                conn.commit()
                conn.close()
                return 'Image uploaded and stored in database'
        elif request.method == 'GET':
                conn = sqlite3.connect('imagesjpg.db')
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM my_imagesjpg")
                rows = cursor.fetchall()
                conn.close()
                images=[]
                for i,row in enumerate(rows):
                        base64_string= row[0].split(',')[1]
                        image_bytes = base64.b64decode(base64_string)
                        filename = f"image_{i + 1}.png"

                        with open(filename, 'wb') as f:
                                f.write(image_bytes)
                        images.append(filename)
                image_tags = []
                for image in images:
                        base64_image = encode_image(image)
                        image_tag = f'<img src="data:image/png;base64,{base64_image}" alt="Image">'
                        #image_tags.append(image_tag)
                        image_tags.append(base64_image)

                html = ' '.join(image_tags)
                return image_tags


from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import LabelEncoder


keys = []
values = []
def extract_keys_values(json_obj, prefix=''):
    for key, value in json_obj.items():
        if isinstance(value, dict):
            nested_keys, nested_values = extract_keys_values(value, prefix + key + '_')
            keys.extend(nested_keys)
            values.extend(nested_values)
        elif isinstance(value, list):
            keys.append(prefix + key)
            values.append(','.join(str(v) for v in value))
        else:
            keys.append(prefix + key)
            values.append(str(value))
    return keys,values




def legalchatbotintentrecognition(data):
    features=[]
    samples=[]
    for d in data:
        for key, value in d.items():
            features.append(key)
            tokens = value
            samples.append(tokens)
    f=[]
    t=[]
    for s in samples:
        k,v= extract_keys_values(s,"")
        f.extend(k)
        t.extend(v)
    l = LabelEncoder()
    encoded_feature = l.fit_transform(f)
    print(encoded_feature)
    feat= np.array(encoded_feature)
    feat= feat.astype(float)
    encoded_samples = l.fit_transform(t)
    samp = np.array(encoded_samples)
    samp = samp.astype(float)
    X_train, X_test, y_train, y_test = train_test_split(feat, samp, test_size=0.25, random_state=42)
    pca = PCA(n_components=1)
    X_train_pca = pca.fit_transform(X_train.reshape(-1, 1))
    print(X_test.shape)
    X_test_pca = pca.transform(X_test.reshape(-1,1))

    svm = SVC()
    svm.fit(X_train_pca, y_train)
    y_pred = svm.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)

    label=LabelEncoder()
    encoder={}
    for i in range(len(f)):
            encoder[f[i]]= encoded_feature[i]

    print("Accuracy:", accuracy)
    print(encoder)
    a= LabelEncoder()
    y= l.inverse_transform(y_pred.astype(int))
    X_train_features = []
    for i in X_test_pca:
        for value in i:
            reshaped_value = value.reshape(-1, 1)
            original_feature = None
            for key, val in encoder.items():
                if val == reshaped_value:
                    original_feature = key
                    break
            if original_feature is not None:
                X_train_features.append(original_feature)
    print(X_train_features)
    return y,X_train_features


from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

@app.route('/kmeans', methods=['POST','GET'])
def pos():
    if request.method == 'POST':
            data = request.get_json()
            df = pd.DataFrame.from_dict(data, orient='index')
            df = df.transpose()
            df.to_csv('customers.csv', index=False, header=True)
            #file = open('customers.csv', 'r')
            #f = csv.reader(file)
            response= json.dumps(data)
            return response
    elif request.method == 'GET':
        df = pd.read_csv('customer_data.csv')
        age = df['Age Group'].values
        gender = df['Gender'].values
        movies = df['Movies'].values
        books = df['Books'].values
        music = df['Music'].values
        sports = df['Sports'].values
        food = df['Food'].values
        label_encoder = LabelEncoder()
        age_encoded = label_encoder.fit_transform(age)
        gender_encoded = label_encoder.fit_transform(gender)
        movies_encoded = label_encoder.fit_transform(movies)
        books_encoded = label_encoder.fit_transform(books)
        music_encoded = label_encoder.fit_transform(music)
        sports_encoded = label_encoder.fit_transform(sports)
        food_encoded = label_encoder.fit_transform(food)
        combined_array = np.vstack(
            (age_encoded, gender_encoded, movies_encoded, books_encoded, music_encoded, sports_encoded, food_encoded)).T
        kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(combined_array)
        labels = kmeans.labels_
        df['Cluster'] = labels
        X = df[['Age Group', 'Gender', 'Movies', 'Books', 'Music', 'Sports', 'Food']]
        y = df['Cluster']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = DecisionTreeClassifier()
        X = np.array(X_train).reshape(-1, 1)
        l = LabelEncoder()
        m = LabelEncoder()
        Xencoded = l.fit_transform(X)
        y_encoded = l.fit_transform(y_train)
        x_reshaped = Xencoded.reshape(-1, 7)
        clf.fit(x_reshaped, y_encoded)
        df2= pd.read_csv('customers.csv')
        json_data = df2.to_json(orient='records')
        datass = json.loads(json_data)
        agegroup= datass[0]['agegroup']
        gender = datass[0]['gender']
        movies= datass[0]['moviePreferences']
        books=datass[0]['books']
        fiction=datass[0]['music']
        sports=datass[0]['sports']
        food = datass[0]['food']
        xtest = np.array([agegroup,gender,movies,books,fiction,sports,food])
        o = m.fit_transform(xtest)
        y_pred = clf.predict(o.reshape(1, -1))
        return jsonify(y_pred.tolist()[0])


import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

def model(input_sequence,output_sequence):
    X_train= np.array(input_sequence)
    y_train=np.array(output_sequence)
    label_encoder = LabelEncoder()
    label_encoder = LabelEncoder()
    X_train_encoded = label_encoder.fit_transform(X_train)
    y_train_encoded = label_encoder.fit_transform(y_train)
    X_train_float = X_train_encoded.astype(float)
    X_train = np.array(X_train_encoded).reshape(len(input_sequence), -1, 1)
    y_train = np.array(y_train_encoded).reshape(len(input_sequence), -1, 1)
    print(X_train_float)

    model= Sequential()
    model.add(LSTM(units=50,return_sequences=True,input_shape=(1,1)))
    model.add(Dropout(0.2))
    model.compile(optimizer='adam',loss='mean_squared_error')
    model.fit(X_train,y_train,epochs=1,batch_size=5)
    return model,y_train_encoded

def chat():
    data1 = [{
        "What are my rights?": {
            "response": "What is the process for registering a company?",
        },
        "How can I protect my intellectual property?": {
            "response": "The legal requirements for starting a business can vary depending on the jurisdiction and the type of business. Some common requirements include registering the business, obtaining necessary licenses and permits, and complying with tax and employment laws.",
            }
    },
        {
            "inputw": {
                "responses": "What is the process for registering a company?",
            },
            "inputq": {
                "responsess": "The legal requirements for starting a business can vary depending on the jurisdiction and the type of business. Some common requirements include registering the business, obtaining necessary licenses and permits, and complying with tax and employment laws.",
            },
            "input": {
                "response": {
                    "registeredName": "Acme Technologies Inc.",

                }
            }
        }

    ]
    intent,features=legalchatbotintentrecognition(data1)
    input_sequence = [
        "response_registeredName",
        "How can I protect my intellectual property?"
    ]
    output_sequence = [
        "What is the process for registering a company?",
        "The legal requirements for starting a business can vary depending on the jurisdiction and the type of business. Some common requirements include registering the business, obtaining necessary licenses and permits, and complying with tax and employment laws."
    ]
    input_query="response_registeredName"
    X=0
    l=LabelEncoder()
    trained_model,y_train_encoded = model(input_sequence, output_sequence)
    y_pred=''
    a=LabelEncoder()

    print(features)
    encoder = {}
    print(y_train_encoded)
    y_pred_decoded= np.array(0.0)
    if input_query in features:
        X= np.array([input_query])
        X_encoded= l.fit_transform(X)
        X_test_reshaped = np.array(X_encoded).reshape(1, -1, 1)
        y_pred= trained_model.predict(X_test_reshaped)
        y_features=[]
        print(y_pred)
        for i in y_pred:
            for value in i:
                reshaped_value = value.reshape(-1, 1)
                print(encoder)
                original_feature = None
                for key, val in encoder.items():
                    if val==reshaped_value:
                        original_feature = key
                        break
                if original_feature is not None:
                    y_features.append(original_feature)
        label_encoder = LabelEncoder()
        label_encoder.fit(output_sequence)
        y_pred_decoded = label_encoder.inverse_transform(y_pred.flatten().round().astype(int))
        print(y_pred_decoded)
    return jsonify(y_pred_decoded.tolist())



data1 = [{
        "What are my rights?": {
            "response": "What is the process for registering a company?",
        },
        "How can I protect my intellectual property?": {
            "response": "The legal requirements for starting a business can vary depending on the jurisdiction and the type of business. Some common requirements include registering the business, obtaining necessary licenses and permits, and complying with tax and employment laws.",
            }
    },
        {
            "inputw": {
                "responses": "What is the process for registering a company?",
            },
            "inputq": {
                "responsess": "The legal requirements for starting a business can vary depending on the jurisdiction and the type of business. Some common requirements include registering the business, obtaining necessary licenses and permits, and complying with tax and employment laws.",
            },
            "input": {
                "response": {
                    "registeredName": "Acme Technologies Inc.",

                }
            }
        }

    ]
intent,features=legalchatbotintentrecognition(data1)
input_sequence = [
        "response_registeredName",
        "How can I protect my intellectual property?"
    ]
output_sequence = [
        "What is the process for registering a company?",
        "The legal requirements for starting a business can vary depending on the jurisdiction and the type of business. Some common requirements include registering the business, obtaining necessary licenses and permits, and complying with tax and employment laws."
    ]
input_query="response_registeredName"
X=0
l=LabelEncoder()
trained_model,y_train_encoded = model(input_sequence, output_sequence)


@app.route('/chatbot',methods=['GET'])
def chatbot():
    input_query = "response_registeredName"
    X = 0
    l = LabelEncoder()
    a = LabelEncoder()
    print(features)
    encoder = {}
    print(y_train_encoded)
    y_pred_decoded = np.array(0.0)
    if input_query in features:
        X = np.array([input_query])
        X_encoded = l.fit_transform(X)
        X_test_reshaped = np.array(X_encoded).reshape(1, -1, 1)
        y_pred = trained_model.predict(X_test_reshaped)
        y_features = []
        print(y_pred)
        for i in y_pred:
            for value in i:
                reshaped_value = value.reshape(-1, 1)
                print(encoder)
                original_feature = None
                for key, val in encoder.items():
                    if val == reshaped_value:
                        original_feature = key
                        break
                if original_feature is not None:
                    y_features.append(original_feature)
        label_encoder = LabelEncoder()
        label_encoder.fit(output_sequence)
        y_pred_decoded = label_encoder.inverse_transform(y_pred.flatten().round().astype(int))
        print(y_pred_decoded)
    return jsonify(y_pred_decoded.tolist())


@app.route('/photos', methods=['POST'])
def upload_video():
        conn = sqlite3.connect('videos.db')
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS my_videos (
                                                                       filename TEXT,
                                                                        path TEXT
                                                                   )''')
        video = request.files['files']
        filename = video.filename
        cursor.execute("INSERT INTO my_videos VALUES (?, ?)", (filename, video.read()))
        conn.commit()
        conn.close()

        return 'Video uploaded and stored in database'


def profilesss():
        name = request.args.get('name')
        age = request.args.get('age')
        email = request.args.get('email')
        print(email)
        data = {
        "name": name,
        "age": age,
        "email": email
        }
        df = pd.DataFrame.from_dict(data, orient='index')
        df = df.transpose()
        df.to_csv('response.csv', index=False, header=True)
        file = open('response.csv', 'r')
        f = csv.reader(file)
        conn = sqlite3.connect('response.db')
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS my_tabless (
                                           column1 TEXT,
                                           column2 INTEGER,
                                           column3 TEXT
                                       )''')
        for row in f:
            cursor.execute("INSERT INTO my_tabless VALUES (?, ?, ?)", row)
        cursor.commit()
        conn.close()
        return "Successful"



def profiles():
        response= {
         "Name": "John Doe",
          "Age": 30,
          "Email": "johndoe@example.com"}
        df= pd.DataFrame.from_dict(response, orient='index')
        df=df.transpose()
        df.to_csv('response.csv', index=False, header=True)
        file = open('response.csv', 'r')
        f = csv.reader(file)
        conn = sqlite3.connect('response.db')
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS my_tables (
                                   column1 TEXT,
                                   column2 INTEGER,
                                   column3 TEXT
                               )''')
        for row in f:
            cursor.execute("INSERT INTO my_tables VALUES (?, ?, ?)", row)
        conn.commit()
        conn.close()
        return "Successful"

def sim2():
        import pandas as pd
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        data = {
                'name': ['John', 'Jane', 'Mike', 'Emily'],
                'interests': ['machine learning, data analysis', 'programming, web development',
                              'machine learning, artificial intelligence', 'data analysis, statistics']
        }

        df = pd.DataFrame(data)

        vectorizer = TfidfVectorizer()

        tfidf_matrix = vectorizer.fit_transform(df['interests'])

        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        def get_top_matches(name, similarity_matrix, df, top_n=3):
                index = df[df['name'] == name].index[0]
                scores = list(enumerate(similarity_matrix[index]))
                scores = sorted(scores, key=lambda x: x[1], reverse=True)
                top_matches = [df.iloc[score[0]]['name'] for score in scores[1:top_n + 1]]
                return top_matches


def sim(name):
        import pandas as pd
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        data = {
                'name': ['John', 'Jane', 'Mike', 'Emily'],
                'interests': ['machine learning, data analysis', 'programming, web development',
                              'machine learning, artificial intelligence', 'data analysis, statistics']
        }
        df= pd.DataFrame(data)
        vectorizer= TfidfVectorizer()
        tfid= vectorizer.fit_transform(df.loc[:,'interests'])
        cos= cosine_similarity(tfid,tfid)
        indexes=[]
        ind= df[df.loc[:,'name']==name].index[0]
        for i in range(len(cos)):
                indices=[]
                for j in cos[i]:
                        if j>0.4:
                                indices.append(np.where(cos[i]==j)[0][0])
                                indexes.append(indices)
        top_matches = []
        for i in indexes[ind]:
                top_matches.append(df.loc[:,'name'][i])
        return top_matches

if __name__ == '__main__':
    app.run(debug=True)
