import json
import pickle
import string
import random
import nltk
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from keras.models import load_model
from nltk.stem import WordNetLemmatizer



lemmatizer = WordNetLemmatizer()

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)
words = []
classes = []
doc_X = []
doc_y = []
def word_classes(words = words,classes = classes):
    # Loop through all the intents
    # Each list to create

    # tokenize each pattern and append tokens to words, the patterns and
    # the associated tag to their associated list
    for intent in intents['intents']:
        for pattern in intent["patterns"]:
            tokens = nltk.word_tokenize(pattern)
            words.extend(tokens)
            doc_X.append(pattern)
            doc_y.append(intent["tag"])

        # add the tag to the classes if it's not there already
        if intent["tag"] not in classes:
            classes.append(intent["tag"])
    # lemmatize all the words in the vocab and convert them to lowercase
    # if the words don't appear in punctuation
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]
    # sorting the vocab and classes in alphabetical order and taking the # set to ensure no duplicates occur
    words = sorted(set(words))
    classes = sorted(set(classes))
    return words,classes

# list for training data
training = []
words,classes = word_classes()
out_empty = [0] * len(classes)
# creating the bag of words model
for idx, doc in enumerate(doc_X):
    bow = []
    text = lemmatizer.lemmatize(doc.lower())
    for word in words:
        bow.append(1) if word in text else bow.append(0)
    # mark the index of class that the current pattern is associated
    # to
    output_row = list(out_empty)
    output_row[classes.index(doc_y[idx])] = 1
    # add the one hot encoded BoW and associated classes to training
    training.append([bow, output_row])
# shuffle the data and convert it to an array
random.shuffle(training)
training = np.array(training, dtype=object)
# split the features and target labels
train_X = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))# list for training data
training = []
out_empty = [0] * len(classes)
# creating the bag of words model
for idx, doc in enumerate(doc_X):
    bow = []
    text = lemmatizer.lemmatize(doc.lower())
    for word in words:
        bow.append(1) if word in text else bow.append(0)
    # mark the index of class that the current pattern is associated to
    output_row = list(out_empty)
    output_row[classes.index(doc_y[idx])] = 1
    # add the one hot encoded BoW and associated classes to training
    training.append([bow, output_row])
# shuffle the data and convert it to an array
random.shuffle(training)
training = np.array(training, dtype=object)
# split the features and target labels
train_X = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# defining some parameters
input_shape = (len(train_X[0]),)
output_shape = len(train_y[0])
epochs = 20

#Clear any existing model in memory
tf.keras.backend.clear_session()

# the deep learning model

#Initialize model
model = Sequential()

# Input layer
model.add(Dense(128, input_shape=input_shape, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))

#Output layer
model.add(Dense(output_shape, activation = "softmax"))

#Defining optimizer
adam = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
#Configuring the model for training
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=["accuracy"])

# Model summary
print(model.summary())

# Training the model
model.fit(x=train_X, y=train_y, epochs=20, verbose=1)
model.save('intent_classifier.h5')

ic_model = load_model('intent_classifier.h5')
#Functions for Chatbot Sessions
def clean_text(text):
  tokens = nltk.word_tokenize(text)
  tokens = [lemmatizer.lemmatize(word) for word in tokens]
  return tokens

def bag_of_words(text, vocab):
  tokens = clean_text(text)
  bow = [0] * len(vocab)
  for w in tokens:
    for idx, word in enumerate(vocab):
      if word == w:
        bow[idx] = 1
  return np.array(bow)

def pred_class(text, vocab, labels):
  bow = bag_of_words(text, vocab)
  result = ic_model.predict(np.array([bow]))[0]
  thresh = 0.2
  y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]

  y_pred.sort(key=lambda x: x[1], reverse=True)
  return_list = []
  for r in y_pred:
    return_list.append(labels[r[0]])
  return return_list

def get_response(intents_list, intents_json):
    print(intents_list)
    tag = intents_list[0]
    #print(tag)
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        #print(i["tag"])
        if i['tag'] == 'issue':
            print(i['tag'])
            result = "Accident"
            break
        elif i['tag'] == tag:
            result = random.choice(i['responses'])
            break
        else:
            result = "You must ask the right questions"
    return result