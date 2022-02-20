import json
import pickle
import nltk
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
#import tensorflow as tf
from intent_classifier import word_classes, pred_class, get_response
#from main_chat import
#print(tf.__version__)
import re,string,unicodedata
from nltk.corpus import stopwords
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('punkt')

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


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence


def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


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

#def predict_class(sentence, model):
    # filter out predictions below a threshold
    #p = bow(sentence, words, show_details=False)
    #X_cat = ['aSummer', 'aMonday', 'Male', 'Mining', '2', '2017', 'Others', 'Employee']
    #X_cat = np.array([0.6])
    #res = (np.asarray(model.predict([tf.stack(np.array([p])), X_cat]))).round()
    #res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    #results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    #results.sort(key=lambda x: x[1], reverse=True)
    #return_list = []
    #for r in results:
    #    return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    #return return_list

#input = 'In moments that the operator of the Jumbo 2, tried energize your equipment to proceed to the installation of 4 split set at intersection 544 of Nv 3300, remove the lock and opening the electric board of 440V and 400A, and when lifting the thermomagnetic key This makes phase to ground - phase contact with the panel shell - producing a flash which reaches the operator causing the injury described.'

def get_final_response(p_ip):
    pred = model.predict(p_ip)
    n = np.array(pred)

    index_max = np.argmax(n)
    return ('Accident level: ' + str(index_max))



def get_chatbot_response(ints, intents_json):
    print(ints)
    tag = ints[0]['intent']
    print(tag)
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == 'issue':
            print(i['tag'])
            result = 'Accident'
            break
        elif i['tag'] == tag:
            result = random.choice(i['responses'])
            break
        else:
            result = "You must ask the right questions"
    return result


def chatbot_response(input):
    #ints = predict_class(msg, model)
    #res = get_chatbot_response(ints, intents)

    #return res
    words, classes = word_classes()
    preds = pred_class(input, words, classes)
    result = get_response(preds, intents)
    if result == "Accident":
        input = pd.DataFrame({'col1': input}, index=[0])
        p_ip = text_processing(input['col1'])
        result = get_final_response(p_ip)
        return result
    return result


if __name__ == "__main__":
    print("BOT : Chat with the bot[Type 'quit' to stop] !")
    print("\nBOT : If answer is not  right[Type '*'] !")
    while True:
        # Reading Input
        input = input("\n\nYou: ")
        # Correcting chat
        if input.lower() == "*":
            print("\nBOT:Please rephrase your question and try again")
        # Stopping Chat
        if input.lower() == "quit":
            break
        # Predicting and printing response
        words,classes = word_classes()
        preds = pred_class(input, words, classes)
        result = get_response(preds, intents)
        print("\nAmelia : ", result)
