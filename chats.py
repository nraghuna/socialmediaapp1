import string
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM,Input,TimeDistributed,Dense,Activation,RepeatVector,Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy
import tensorflow as tf


import pandas as pd
from flask import Flask, jsonify, request
import requests
import pandas as pd
from flask_cors import CORS
import json
import sqlite3
import csv
app = Flask(__name__)
import numpy as np
CORS(app)

def cleansentence(sentence):
    lower_case_sent= sentence.lower()
    string_punctuation= string.punctuation + "¡" + '¿'
    clean_sentence = lower_case_sent.translate(str.maketrans('', '', string_punctuation))
    return clean_sentence

def tokenize(sentences):
    text_tokenizer= Tokenizer()
    text_tokenizer.fit_on_texts(sentences)
    return text_tokenizer.texts_to_sequences(sentences),text_tokenizer


def find_word_index(input_sequence,tokenizer, word):
    index_to_words = {idx: word for word, idx in tokenizer.word_index.items()}
    word_index = tokenizer.word_index
    words = tokenizer.word_index.items()
    related_word = index_to_words[word_index[word]]
    c=tokenizer.word_index[related_word]
    return c

def logits_to_sentence(logits, tokenizer,i):
    index_to_words = {idx: word for word, idx in tokenizer.word_index.items()}
    index_to_words[0] = '<empty>'

    predictions = np.argmax(logits[0:i], axis=1)

    words = [index_to_words[prediction] for prediction in predictions[0][0:i]]
    sentence = ' '.join(words)
    return sentence
def chat(h,t):
    input_sentences= h
    output_sentences=t
    input_tokenized, input_tokenizer = tokenize(input_sentences)
    output_tokenized, output_tokenizer = tokenize(output_sentences)
    input_len = len(max(input_tokenized, key=len))
    output_len = int(len(max(output_tokenized, key=len)))
    output_vocab = len(output_tokenizer.word_index) + 1
    input_vocab = len(input_tokenizer.word_index) + 1
    input_pad_sentence = pad_sequences(input_tokenized, input_len, value=0)
    output_pad_sentence = pad_sequences(output_tokenized, output_len, value=0)
    #input_pad_sentence = input_pad_sentence.reshape(*input_pad_sentence.shape, 1)
    input_pad_sentence = input_pad_sentence.reshape(input_pad_sentence.shape[0], input_pad_sentence.shape[1])
    output_pad_sentence = output_pad_sentence.reshape(*output_pad_sentence.shape, 1)
    input_sequence = Input(shape=(input_len,))
    embedding = Embedding(input_dim=input_vocab, output_dim=128, )(input_sequence)
    encoder = LSTM(64, return_sequences=False)(embedding)
    r_vec = RepeatVector(output_len)(encoder)
    decoder = LSTM(64, return_sequences=True, dropout=0.2)(r_vec)
    logits = TimeDistributed(Dense(output_vocab))(decoder)
    enc_dec_model = Model(input_sequence, Activation('softmax')(logits))
    enc_dec_model.compile(loss=sparse_categorical_crossentropy,
                          optimizer=Adam(1e-3),
                          metrics=['accuracy'])
    enc_dec_model.summary()
    enc_dec_model.fit(input_pad_sentence, output_pad_sentence)
    i=find_word_index(input_pad_sentence,input_tokenizer,"advantages")
    d=logits_to_sentence(enc_dec_model.predict(input_pad_sentence), output_tokenizer,i)
    return d


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
    return f,t


if __name__ == '__main__':
    h,t=legalchatbotintentrecognition(data)
    c= chat(h,t)
    print(c)