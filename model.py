import joblib
import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
from nltk import word_tokenize
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet, stopwords
import joblib as jb
stemmer = PorterStemmer()
intab = string.punctuation
outtab = "                                "
trantab = str.maketrans(intab, outtab)

def negation_handler(text):
    tokens = word_tokenize(text)
    handled = ''
    for i in range(len(tokens)):
        if (tokens[i] == 'not') | (tokens[i] == 'n\'t'):
            try:
                index = i + 1
                if (tokens[i + 1] == 'amazing') | (tokens[i + 1] == 'great'):
                    tokens[i + 1] = 'bad'
                    continue
                elif (tokens[i + 1] == 'very') | (tokens[i + 1] == 'so'):
                    index = i+2
                f = False
                for syn in wordnet.synsets(tokens[index]):
                    if f:
                        break
                    for l in syn.lemmas():
                        if l.antonyms():
                            tokens[i + 1] = l.antonyms()[0].name()
                            f = True
                            break
            except:
                continue
        else:
            handled += tokens[i] + ' '
    return handled

def new_scores(df, text, sentiment):
    df = df.reset_index(drop=True)
    df = df.dropna(subset=[text, sentiment])
    df = df.reset_index(drop=True)
    map = {1: 0, 2: 0, 4: 1, 5: 1, 'positive': 1, 'negative': 0}
    df[sentiment] = df[sentiment].map(map)
    groups = df.groupby(sentiment)
    df_pos = groups.get_group(1)
    df_neg = groups.get_group(0)
    df = pd.concat([df_pos, df_neg], ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.rename(columns={text: "Text"})
    df = df.rename(columns={sentiment: "Score"})
    df = df[['Text', 'Score']]
    return df

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return ' '.join(stems)

def remove_stops(text):
    tokens = nltk.word_tokenize(text)
    stops = []
    with open('stops.txt', 'r') as file:
        for readline in file:
            stops.append(readline.strip())
    for i in range(len(tokens)):
        for s in stops:
            if tokens[i] == s:
                tokens[i] = ''
    return ' '.join(tokens)

messages = pd.read_csv('./Reviews.csv')
messages = messages[messages['Score'] != 3]
messages = new_scores(messages, 'Text', 'Score')
messages['Text'] = messages['Text']
Summary = messages['Text']
Score = messages['Score']
X_train, X_test, y_train, y_test = train_test_split(Summary, Score, test_size=0.2, random_state=42)

# --- Training set
print('we are pre processing training set')
corpus = []

for text in X_train:
    if type(text) == str:
        text = text.lower()
        text = negation_handler(text)
        text = text.translate(trantab)
        text = remove_stops(text)
        text = tokenize(text)
        corpus.append(text)

count_vect = CountVectorizer(min_df=0.3,max_df=0.7)
X_train_counts = count_vect.fit_transform(corpus)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
words_set = tfidf_transformer.get_feature_names()
f = open("myfile.txt", "x")
f.close()
f = open("myfile.txt", "w")
for word in words_set:
    f.write(word,' ')
print(words_set)
# --- Test set
print('we are pre processing testing set')

test_set = []
for text in X_test:
    if type(text) == str:
        text = text.lower()
        text = negation_handler(text)
        text = text.translate(trantab)
        text = tokenize(text)
        text = remove_stops(text)
        test_set.append(text)

X_new_counts = count_vect.transform(test_set)
X_test_tfidf = tfidf_transformer.transform(X_new_counts)
joblib.dump(count_vect, 'tf_idf2.joblib')
from sklearn import linear_model

print('we are pre fitting training set on linear regression')
logreg = linear_model.LogisticRegression(solver='liblinear', random_state=42, max_iter=1000, C=1e5)
model = logreg.fit(X_train_tfidf, y_train)
joblib.dump(model, 'Logistic Regression2 93.5%.joblib')
predictions = logreg.predict(X_test_tfidf)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print(accuracy_score(y_test, predictions))
