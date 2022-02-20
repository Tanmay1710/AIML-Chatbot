import json
import pickle

import nltk
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer

import tensorflow as tf; print(tf.__version__)
import re,string,unicodedata

from nltk.corpus import stopwords

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

bot_name = "Amelia"
lemmatizer = WordNetLemmatizer()

model = load_model('chatbot_model_m3.h5')
import json
import random

intents = json.loads(open('intents.json', encoding='utf-8').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

def text_processing(input):
    stop_words = stopwords.words('english')

    ## Preprocessing the text

    ##Changing to lower case and substituiting numeric and special characters with nothing

    input = input.apply(lambda x: x.lower())
    input = input.apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))
    # input = input.lower()

    ## Removing Stop words

    stop = set(stopwords.words('english'))
    punctuation = list(string.punctuation)
    stop.update(punctuation)

    # Removing the stopwords from text
    def remove_stopwords(text):
        final_text = []
        for i in text.split():
            if i.strip().lower() not in stop:
                final_text.append(i.strip())
        return " ".join(final_text)

    # Apply function on review column
    input = input.apply(remove_stopwords)

    X_text = input
    # The first step in word embeddings is to convert the words into thier corresponding numeric indexes.
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_text)

    X_text = tokenizer.texts_to_sequences(X_text)

    # Sentences can have different lengths, and therefore the sequences returned by the Tokenizer class also consist of variable lengths.
    # We need to pad the our sequences using the max length.
    vocab_size = len(tokenizer.word_index) + 1
    print("vocab_size:", vocab_size)

    maxlen = 100

    X_text = pad_sequences(X_text, padding='post', maxlen=maxlen)
    return X_text

def get_final_response(p_ip):
    pred = model.predict(p_ip)
    n = np.array(pred)

    index_max = np.argmax(n)
    return ('Accident level: ' + str(index_max))

def chatbot_response(sentence):
    sentence = text_processing(sentence)
    res= get_final_response(sentence)
    return res



if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break

        # resp = get_response(sentence)
        resp = chatbot_response(sentence)
        print(resp)

