from data_preparation import text_tokenizer, token
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from matplotlib import pyplot as plt
from tabulate import tabulate

data = pd.read_csv('hp.txt', sep='=', usecols=["book_list", "value"])
data = data['value'][:10]

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
print("Most common tokens:\n", pd.DataFrame(common_tockens), "\n")

common_tockens.plot(kind='barh', x='Token', y='Występowanie')
print(tabulate(common_tockens, headers='keys', tablefmt='psql'))
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
print("Key tokens:\n", res_key_tokens, "\n")

res_key_tokens.plot(kind='barh', x='Token', y='Wartość')
print(tabulate(res_key_tokens, headers='keys', tablefmt='psql'))
plt.show()

