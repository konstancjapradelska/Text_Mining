import pandas as pd
import data_preparation
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from matplotlib import pyplot as plt

# przygotowanie danych
data_org = pd.read_csv('hp.txt', sep='=', usecols=["book_list", "value"])
data_org['value'] = data_org['value'].str[:1000]

data = data_org["value"]
data_type = data_org["book_list"]

data_train, data_test, type_train, type_test = train_test_split(data, data_type, test_size=0.3, random_state=1)

# wektoryzacja
vectorizer = CountVectorizer(tokenizer=data_preparation.text_tokenizer)
data_train_transform = vectorizer.fit_transform(data_train)
data_test_transform = vectorizer.transform(data_test)

# Drzewo decyzyjne
print("\nDrzewo decyzyjne")
DecisionTree = DecisionTreeClassifier()
DecisionTree = DecisionTree.fit(data_train_transform, type_train)
print("Dokładność w zbiorze uczącym: ", (DecisionTree.score(data_train_transform, type_train)) * 100)
print("Dokładność w zbiorze testowym: ", (DecisionTree.score(data_test_transform, type_test)) * 100)
plot_confusion_matrix(DecisionTree, data_test_transform, type_test)
plt.title('Confusion matrix dla drzewa decyzyjnego')
plt.show()

# Las losowy
print("\nLas losowy")
RandomForest = RandomForestClassifier(n_estimators=5, random_state=2)
RandomForest = RandomForest.fit(data_train_transform, type_train)
print("Dokładność w zbiorze uczącym: ", (RandomForest.score(data_train_transform, type_train)) * 100)
print("Dokładność w zbiorze testowym: ", (RandomForest.score(data_test_transform, type_test)) * 100)
plot_confusion_matrix(RandomForest, data_test_transform, type_test)
plt.title('Confusion matrix dla lasu losowego')
plt.show()

# SVM
print("\nSVM")
svm = SVC()
svm = svm.fit(data_train_transform, type_train)
print("Dokładność w zbiorze uczącym: ", (svm.score(data_train_transform, type_train)) * 100)
print("Dokładność w zbiorze testowym: ", (svm.score(data_test_transform, type_test)) * 100)
plot_confusion_matrix(svm, data_test_transform, type_test)
plt.title('Confusion matrix dla SVM')
plt.show()


# AdaBoost
print("\nAdaBoost")
AdaBoost = AdaBoostClassifier()
AdaBoost = AdaBoost.fit(data_train_transform, type_train)
print("Dokładność w zbiorze uczącym: ", (AdaBoost.score(data_train_transform, type_train)) * 100)
print("Dokładność w zbiorze testowym: ", (AdaBoost.score(data_test_transform, type_test)) * 100)
plot_confusion_matrix(AdaBoost, data_test_transform, type_test)
plt.title('Confusion matrix dla AdaBoost')
plt.show()

# Bagging
print("\nBagging")
Bagging = BaggingClassifier()
Bagging = Bagging.fit(data_train_transform, type_train)
print("Dokładność w zbiorze uczącym: ", (Bagging.score(data_train_transform, type_train)) * 100)
print("Dokładność w zbiorze testowym: ", (Bagging.score(data_test_transform, type_test)) * 100)
plot_confusion_matrix(Bagging, data_test_transform, type_test)
plt.title('Confusion matrix dla BaggingClassifier')
plt.show()
