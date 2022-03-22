import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')



string = 'ad Lorem  Ipsum is simply and also dummy ay text of the printing and typesetting industry.' \
         ' Lorem Ipsum has been the industry\'s standard dummy text ever since the 1500s,' \
         ' when an unknown printer bela took a galley of type and scrambled it to make a type specimen book.' \
         ' It has survived not ;) only five centuries, but also the leap into electronic typesetting, ' \
         'remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset ' \
         'sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like ' \
         'Aldus PageMaker including versions of Lorem Ipsum. :)'
temp = string.split()


def stemm_funtion(string: str) -> list:
    stem_list: list = []
    porter = PorterStemmer()
    for word in temp:
        stem_list.append(porter.stem(word))
    return stem_list


temp = stemm_funtion(temp)
print(temp)


def clear_text(string: str):
    emoticons = re.findall(':\)|;\)|;\(|:>|:<|;<|:-\)|;-\)', string)
    string = re.sub(r'\d', '', string)
    string = re.sub('[<>/]', '', string)
    string = re.sub('[,.;:]', '', string)
    string = re.sub(r'\s+', ' ', string)
    string = string.lower()
    string = string + str(emoticons)
    return string


string = clear_text(string)
print(string)

res = string.split()


def stopword_rem(string: str):
    for word in res:
        if word in stopwords.words():
            string = string.replace(word, '')
    return string


print(stopword_rem(string))
