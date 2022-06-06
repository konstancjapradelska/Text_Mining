import pandas as pd
import lab_4
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# przygotowanie danych
data_True = pd.read_csv('True.csv', usecols=['title'], nrows=100)
data_Fake = pd.read_csv('Fake.csv', usecols=['title'], nrows=100)
data_True['type'] = 'True'
data_Fake['type'] = 'Fake'
data_org = pd.concat([data_True, data_Fake])

# zadanie 1
data = data_org['title']
data_type = data_org['type']
data_train, data_test, type_train, type_test = train_test_split(data, data_type, test_size=0.3, random_state=1)

# zadanie 2
vectorizer = CountVectorizer(tokenizer=lab_4.text_tokenizer)
data_train_transform = vectorizer.fit_transform(data_train)
data_test_transform = vectorizer.transform(data_test)

# zadanie 3 A
print("\nDrzewo decyzyjne")
DecisionTree = DecisionTreeClassifier()
DecisionTree = DecisionTree.fit(data_train_transform, type_train)
print("Dokładność w zbiorze uczącym: ", (DecisionTree.score(data_train_transform, type_train)))
print("Dokładność w zbiorze testowym: ", (DecisionTree.score(data_test_transform, type_test)))

# zadanie 3 B
print("\nLas losowy")
RandomForest = RandomForestClassifier(n_estimators=4, random_state=2)
RandomForest = RandomForest.fit(data_train_transform, type_train)
print("Dokładność w zbiorze uczącym: ", (RandomForest.score(data_train_transform, type_train)))
print("Dokładność w zbiorze testowym: ", (RandomForest.score(data_test_transform, type_test)))

# zadanie 3 C
print("\nSVM")
svm = SVC()
svm = svm.fit(data_train_transform, type_train)
print("Dokładność w zbiorze uczącym: ", (svm.score(data_train_transform, type_train)))
print("Dokładność w zbiorze testowym: ", (svm.score(data_test_transform, type_test)))

# zadanie 3 D
print("\nAdaBoost")
AdaBoost = AdaBoostClassifier()
AdaBoost = AdaBoost.fit(data_train_transform, type_train)
print("Dokładność w zbiorze uczącym: ", (AdaBoost.score(data_train_transform, type_train)))
print("Dokładność w zbiorze testowym: ", (AdaBoost.score(data_test_transform, type_test)))

# zadanie 3 E
print("\nBagging")
Bagging = BaggingClassifier()
Bagging = Bagging.fit(data_train_transform, type_train)
print("Dokładność w zbiorze uczącym: ", (Bagging.score(data_train_transform, type_train)))
print("Dokładność w zbiorze testowym: ", (Bagging.score(data_test_transform, type_test)))
