from data_preparation import text_tokenizer, clear_text, stemm_funtion, stopword_rem
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from matplotlib import pyplot as plt
from tabulate import tabulate
from wordcloud import *
from typing import Dict

# zadanie 1
# wczytanie danych
data_org = pd.read_csv('tweets_airline.csv', sep=',', usecols=["airline_sentiment", "text"])
data = data_org['text']  #[:1000] do użycia w przypadku zbyt długiego czasu obliczania

# zadanie 3
# wordcloud
temp = clear_text(str(data_org))
temp = stemm_funtion(temp)
temp = stopword_rem(temp)

bow: Dict[str, float] = {}

for word in temp:
    if word not in bow.keys():
        bow[word] = 1
    elif word in bow.keys():
        bow[word] += 1

wc = WordCloud()
wc.generate_from_frequencies(bow)
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()

## wektoryzacja
vectorizer = CountVectorizer(tokenizer=text_tokenizer)
x_transform = vectorizer.fit_transform(data)
column_names = vectorizer.get_feature_names_out()
array_xtransform = x_transform.toarray()

## Most common tokens
most_common_tokens = []
token_occurrences_amount = []
column_sums = np.sum(array_xtransform, axis=0)
most_occurring_index = np.argpartition(column_sums, -10)[-10:]

for i in np.nditer(most_occurring_index):
    most_common_tokens.append(column_names[i])
    token_occurrences_amount.append(column_sums[i])

common_tockens = {'Token': most_common_tokens, 'Występowanie': token_occurrences_amount}
common_tockens = pd.DataFrame(common_tockens)

common_tockens.plot(kind='barh', x='Token', y='Występowanie')
print(tabulate(common_tockens, headers='keys', tablefmt='psql'))
plt.title('Najczesciej wystepujace tokeny')
plt.show()

## Key tokens
key_tokens = []
key_tokens_value = []
token_column_sums = np.sum(array_xtransform, axis=0)
key_tokens_index = np.argpartition(token_column_sums, -10)[-10:]

for i in np.nditer(key_tokens_index):
    key_tokens.append((column_names[i]))
    key_tokens_value.append(token_column_sums[i])

res_key_tokens = {'Token': key_tokens, 'Wartość': key_tokens_value}
res_key_tokens = pd.DataFrame(res_key_tokens)

res_key_tokens.plot(kind='barh', x='Token', y='Wartość')
print(tabulate(res_key_tokens, headers='keys', tablefmt='psql'))
plt.title('Najwazniejsze tokeny')
plt.show()
