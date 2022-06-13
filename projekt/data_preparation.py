import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize

stop_words = stopwords.words("english")


def clear_text(string: str) -> str:
    emoticons = re.findall(r'[:|;][-]?[)|(|<>]', string)
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
    string = string.replace('want', '')
    string = string.replace('around', '')
    string = string.replace('would', '')
    string = string.replace('though', '')
    string = string.replace('think', '')
    string = string.replace('time', '')
    string = string.replace('face', '')
    string = string.replace('hand', '')
    string = string.replace('come', '')
    string = string.replace('still', '')
    string = string.replace('voice', '')
    string = string.replace('seem', '')
    return string


def stopword_rem(text: str) -> list:
    return [w for w in text if not w.lower() in stop_words]


def stemm_funtion(word: str) -> str:
    ps = PorterStemmer()
    return ps.stem(word)


def text_tokenizer(text: str):
    temp = clear_text(text)
    temp = word_tokenize(temp)
    temp = stopword_rem(temp)
    temp_list = []
    for i in temp:
        temp_list.append(i)
    return [stemm_funtion(w) for w in temp_list if len(w) > 3]

def token(x):
    vectorizer = CountVectorizer(tokenizer=text_tokenizer)
    data_transform = vectorizer.fit_transform(x)
    return data_transform
