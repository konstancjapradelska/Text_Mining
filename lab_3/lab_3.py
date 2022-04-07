import re
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import *

stemmer = SnowballStemmer("english")

data_Fake = pd.read_csv('Fake.csv')
data_True = pd.read_csv('True.csv')
data_Fake = data_Fake['title']
data_True = data_True['title']


def clear_text(string: str) -> str:
    emoticons = re.findall(':\)|;\)|;\(|:>|:<|;<|:-\)|;-\)', string)
    string = re.sub(r'\d', '', string)
    string = re.sub('[<>/]', '', string)
    string = re.sub('[,.;:]', '', string)
    string = re.sub(r'\s+', ' ', string)
    string = string.lower()
    string = string + str(emoticons)
    return string


temp = clear_text(str(data_Fake))
print(temp)


def stemm_funtion(string: str) -> list:
    stem_list = []
    porter = PorterStemmer()
    stream = string.split(' ')
    for word in stream:
        stem_list.append(porter.stem(word))
    return stem_list


temp = stemm_funtion(temp)
print(temp)


def stopword_rem(text: list) -> list:
    return [word for word in text if word not in stopwords.words()]


temp = stopword_rem(temp)
print(temp)


bow: Dict[str, float] = {}

for word in temp:
    if word not in bow.keys():
        bow[word] = 1
    elif word in bow.keys():
        bow[word] += 1

print(bow)

wc = WordCloud()
wc.generate_from_frequencies(bow)
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()

