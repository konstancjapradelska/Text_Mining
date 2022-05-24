from lab_4 import text_tokenizer, token
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

data_Fake = pd.read_csv('Fake.csv')
data_Fake = data_Fake['text'][:10]

data_True = pd.read_csv('True.csv')
data_True = data_True['text'][:10]

vectorizer = CountVectorizer(tokenizer=text_tokenizer)
x_transform = vectorizer.fit_transform(data_Fake)
column_names = vectorizer.get_feature_names_out()
array_xtransform = x_transform.toarray()

## zad 1
one_transform = token(data_True[:1])
multiple_transform = token(data_True[:4])
print("Zadanie 1:\n")
print("Jeden:\n ", one_transform.toarray())
print("Wiele:\n ", multiple_transform.toarray())
print("Przy przekazaniu 1 dokumentu w macierzy nie zobaczymy zer.\n")

## zad 2
most_common_tokens = []
token_occurrences_amount = []
column_sums = np.sum(array_xtransform, axis=0)
most_occurring_index = np.argpartition(column_sums, -10)[-10:]

for i in np.nditer(most_occurring_index):
    most_common_tokens.append(column_names[i])
    token_occurrences_amount.append(column_sums[i])

data = {'Token': most_common_tokens, 'Występowanie': token_occurrences_amount}
print("Zadanie 2:\n", pd.DataFrame(data),"\n")

## zad 3
key_tokens = []
key_tokens_value = []
token_column_sums = np.sum(array_xtransform, axis=0)
key_tokens_index = np.argpartition(token_column_sums, -10)[-10:]

for i in np.nditer(key_tokens_index):
    key_tokens.append((column_names[i]))
    key_tokens_value.append(token_column_sums[i])

data = {'Token': key_tokens, 'Wartość': key_tokens_value}
print("Zadanie 3:\n", pd.DataFrame(data),"\n")

##zad 4
top_documents = []
row_sums = np.sum(array_xtransform, axis=1)
top_documents_index = np.argpartition(row_sums, -10)[-10:]

for i in np.nditer(top_documents_index):
    top_documents.append(data_Fake[i])
print("Zadanie 4:\n", top_documents)
