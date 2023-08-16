import nltk
import time
import random
import string
import numpy as np
import pandas as pd
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


train = pd.read_csv("https://raw.githubusercontent.com/eliotjmartin/uodsc-club/main/twitter_train.csv")
train = train.dropna()

#Preproccessing

def tokenizer(sentence):
    # remove punctuation from sentence
    sentence = ''.join(
        char for char in sentence if char not in string.punctuation
    )
    # tokenizing the sentence
    tokens = nltk.word_tokenize(sentence)
    return [token.lower() for token in tokens]

stop_words = set(stopwords.words('english'))
'do' in stop_words, 'when' in stop_words

def stopword_destroyer(tokens):
    new = []
    for x in tokens:
        if x not in stop_words:
            new.append(x)

    return new

#Stemming Process

stemmer = PorterStemmer()

def stemmerizer(tokens):
    new = []
    for x in tokens:
        y = stemmer.stem(x)
        new.append(y)
    return new


def preprocess(sentence):
    """
    This function takes a sentence as input and performs various text preprocessing steps on it,
    including removing punctuation, stop words, and stemming each word in the sentence.
    """
    # tokenizing the sentence
    tokens = tokenizer(sentence)

    # removing stop words
    tokens = stopword_destroyer(tokens)

    # stemming each word in the sentence
    tokens = stemmerizer(tokens)

    # return the preprocessed sentence as a list of words
    return tokens


def bag_of_words(tokenized_sentence, map):
    """
    Create a bag of words representation for a given tokenized sentence.
    """
    # initialize the bag with zeros for each word in the vocabulary
    bag = np.zeros(len(map), dtype=np.int8)

    # update the bag with 1 for each word in the sentence that exists in the vocabulary

    for token in tokenized_sentence:
        try:
            bag[map[token]] = 1
        except:
            continue

    return bag

#Loading and Preprocessing

def fullDataPrep(df, map=None, all_words_list=None):
    # build a set of all words if map is none
    if map is None:
        all_words = {}

    preprocessed_list = []
    for sentence in df['text']:
        preprocessed = preprocess(sentence)
        preprocessed_list.append(preprocessed)

        if map is None:
            for token in preprocessed:
                if token in all_words:
                    all_words[token] += 1
                else:
                    all_words[token] = 0

    if map is None:
        keys_to_delete = []
        for key, value in all_words.items():
            if value <= 5:
                keys_to_delete.append(key)

        for key in keys_to_delete:
            del all_words[key]

    # order set by making it a sorted list if map is none
    if map is None:
        all_words_list = sorted(list(all_words.keys()))

    # create a mapping from words to corresponding index if map is
    # none
    # this is an optimization...
    if map is None:
        map = {}
        for i in range(len(all_words_list)):
            word = all_words_list[i]
            map[word] = i

    # build new dataframe with bow repr
    bow_array = []
    for sentence in preprocessed_list:
        row = bag_of_words(sentence, map)
        bow_array.append(row)

    bow_array = np.array(bow_array)

    bow_dict = {}
    for i in range(len(all_words_list)):
        word = all_words_list[i]
        bow_dict[word] = bow_array[:, i]

    return pd.DataFrame(bow_dict), map, all_words_list

#Test
#Contents of all words

new_train, map, all_words = fullDataPrep(train)

#Create and Train Model

X_train, y_train = new_train, train['sentiment']

def y_encode(row):
      """
      encode the target column into integers we can work with
      """
      if row == 'negative':
        return 0
      elif row =='neutral':
        return 1
      return 2

y_train = pd.Series(y_train.apply(y_encode))

#Fit and Predict Model

lr = LogisticRegression(max_iter=1000, random_state=42)
reg = lr.fit(X_train, y_train)

test = pd.read_csv("https://raw.githubusercontent.com/eliotjmartin/uodsc-club/main/twitter_test.csv")
test = test.dropna()

new_test, map, all_words = fullDataPrep(test, map, all_words)

#Test Data
X_test, y_test = new_test, test['sentiment']
y_test = pd.Series(y_test.apply(y_encode))

from sklearn.metrics import accuracy_score
# Predict the labels for the test data
y_pred = reg.predict(X_train)

# Calculate accuracy
accuracy = accuracy_score(y_pred, y_test)
print("Accuracy:", accuracy)