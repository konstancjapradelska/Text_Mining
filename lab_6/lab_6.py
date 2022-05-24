import lab_5
from matplotlib import pyplot as plt
from tabulate import tabulate


lab_5.data_zad2.plot(kind='barh', x='Token', y='Występowanie')
print(tabulate(lab_5.data_zad2, headers='keys', tablefmt='psql'))
plt.show()

lab_5.data_zad3.plot(kind='barh', x='Token', y='Wartość')
print(tabulate(lab_5.data_zad3, headers='keys', tablefmt='psql'))
plt.show()

print(lab_5.top_documents)
