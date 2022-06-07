import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer


def clear_text(string: str) -> str:
    emoticons = re.findall(':\)|;\)|;\(|:>|:<|;<|:-\)|;-\)', string)
    string = re.sub(r'\d', '', string)
    string = re.sub('[<>/]', '', string)
    string = re.sub('[,.;:]', '', string)
    string = re.sub(r'\s+', ' ', string)
    string = string.lower()
    string = string + str(emoticons)
    string = string.replace('look', '')
    string = string.replace('said', '')
    string = string.replace('back', '')
    string = string.replace('like', '')
    string = string.replace('know', '')
    string = string.replace('could', '')
    string = string.replace('very', '')
    return string


def stemm_funtion(string: str) -> list:
    stem_list = []
    porter = PorterStemmer()
    stream = string.split(' ')
    for word in stream:
        stem_list.append(porter.stem(word))
    return stem_list


def stopword_rem(text: list) -> list:
    return [word for word in text if word not in stopwords.words()]


def text_tokenizer(text: str) -> list:
    result = []
    dt_fk = clear_text(text)
    dt_fk = stemm_funtion(dt_fk)
    dt_fk = stopword_rem(dt_fk)
    for word in dt_fk:
        if len(word) > 3:
            result.append(word)
    return result


def token(x):
    vectorizer = CountVectorizer(tokenizer=text_tokenizer)
    data_transform = vectorizer.fit_transform(x)
    return data_transform
