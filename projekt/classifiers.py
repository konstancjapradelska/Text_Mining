import pandas as pd
import data_preparation
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, classification_report, accuracy_score
from matplotlib import pyplot as plt

# przygotowanie danych
data_org = pd.read_csv('hp4.txt', sep='=', usecols=["book_list", "value"])
data_org = data_org.explode('value')
data = data_org["value"]
data_type = data_org["book_list"]

data_train, data_test, type_train, type_test = train_test_split(data, data_type, test_size=0.3, random_state=2)

# wektoryzacja
vectorizer = CountVectorizer(tokenizer=data_preparation.text_tokenizer)
data_train_transform = vectorizer.fit_transform(data_train.values.astype('U'))
data_test_transform = vectorizer.transform(data_test.values.astype('U'))

# Drzewo decyzyjne
print("\nDrzewo decyzyjne")
DecisionTree = DecisionTreeClassifier()
DecisionTree = DecisionTree.fit(data_train_transform, type_train)
print("Dokładność w zbiorze uczącym: ", (DecisionTree.score(data_train_transform, type_train)) * 100)
print("Dokładność w zbiorze testowym: ", (DecisionTree.score(data_test_transform, type_test)) * 100)
plot_confusion_matrix(DecisionTree, data_test_transform, type_test)
plt.title('Confusion matrix dla drzewa decyzyjnego')
plt.show()

predict = DecisionTree.predict(data_test_transform)
report = classification_report(type_test, predict)
print("Classification report:")
print(report)


# Las losowy
print("\nLas losowy")
RandomForest = RandomForestClassifier(n_estimators=5, random_state=2)
RandomForest = RandomForest.fit(data_train_transform, type_train)
print("Dokładność w zbiorze uczącym: ", (RandomForest.score(data_train_transform, type_train)) * 100)
print("Dokładność w zbiorze testowym: ", (RandomForest.score(data_test_transform, type_test)) * 100)
plot_confusion_matrix(RandomForest, data_test_transform, type_test)
plt.title('Confusion matrix dla lasu losowego')
plt.show()
predict = RandomForest.predict(data_test_transform)
report = classification_report(type_test, predict)
print("Classification report:")
print(report)



# SVM
print("\nSVM")
svm = SVC()
svm = svm.fit(data_train_transform, type_train)
print("Dokładność w zbiorze uczącym: ", (svm.score(data_train_transform, type_train)) * 100)
print("Dokładność w zbiorze testowym: ", (svm.score(data_test_transform, type_test)) * 100)
plot_confusion_matrix(svm, data_test_transform, type_test)
plt.title('Confusion matrix dla SVM')
plt.show()
predict = svm.predict(data_test_transform)
report_svm = classification_report(type_test, predict)
print("Classification report:")
print(report_svm)

# AdaBoost
print("\nAdaBoost")
AdaBoost = AdaBoostClassifier()
AdaBoost = AdaBoost.fit(data_train_transform, type_train)
print("Dokładność w zbiorze uczącym: ", (AdaBoost.score(data_train_transform, type_train)) * 100)
print("Dokładność w zbiorze testowym: ", (AdaBoost.score(data_test_transform, type_test)) * 100)
plot_confusion_matrix(AdaBoost, data_test_transform, type_test)
plt.title('Confusion matrix dla AdaBoost')
plt.show()
predict = AdaBoost.predict(data_test_transform)
report = classification_report(type_test, predict)
print("Classification report:")
print(report)

# Bagging
print("\nBagging")
Bagging = BaggingClassifier()
Bagging = Bagging.fit(data_train_transform, type_train)
print("Dokładność w zbiorze uczącym: ", (Bagging.score(data_train_transform, type_train)) * 100)
print("Dokładność w zbiorze testowym: ", (Bagging.score(data_test_transform, type_test)) * 100)
plot_confusion_matrix(Bagging, data_test_transform, type_test)
plt.title('Confusion matrix dla BaggingClassifier')
plt.show()
predict = Bagging.predict(data_test_transform)
report = classification_report(type_test, predict)
print("Classification report:")
print(report)

print("Najlepszy klasyfikator: SVM")
print(report_svm,"\n Precision: Precyzja określa jaki procent wszystkich przypadków zaklasyfikowanych jako pozytywne był poprawny. Dla analizowaych danych precyzja waha się od 58% do 65%","\n Recall: Recall sprawdza jaki procent wszystkich przypadków, które były rzeczywiście pozytywne, został sklasyfikowany poprawnie. Dla analizowanych danych wynii wahają się d 44% do 69%.","\n F1 - score:Wynik F1 jest ważoną średnią harmoniczną precyzji i przywołania, tak że najlepszy wynik to 1,0, a najgorszy to 0,0. Dla analizowanych danych wyniki wahają się od 0.5 do 0.66.","\n Support:Wsparcie to liczba rzeczywistych wystąpień danej klasy w określonym zbiorze danych. W analizowanych danych 3 klasy mają mniej więcej równą liczbę wystąpień w danych - 1 klasa dstaja ta sama klasa i dlatego przejawia najgorsze rezultaty w pozostałych analizowanych wskaźnikach.")
