import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.discriminant_analysis
import sklearn.preprocessing

arquivo = pd.read_csv('cancer-dados.csv')

if 'Unnamed: 32' in arquivo.columns:
    arquivo = arquivo.drop(['Unnamed: 32'], axis=1)

arquivo['diagnosis'] = arquivo['diagnosis'].map({'B': 0, 'M': 1})

X = arquivo.drop(['id', 'diagnosis'], axis=1)
y = arquivo['diagnosis']

escalador = sklearn.preprocessing.StandardScaler()
xPadronizado = escalador.fit_transform(X)

lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(n_components=1)
xLda = lda.fit_transform(xPadronizado, y)

plt.figure(figsize=(8,6))
plt.scatter(xLda[y == 0], np.zeros_like(xLda[y == 0]), alpha=0.8, color='blue', label='Benigno')
plt.scatter(xLda[y == 1], np.zeros_like(xLda[y == 1]), alpha=0.8, color='red', label='Maligno')

plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA - Separação das Classes (Câncer)')
plt.xlabel('LDA Componente 1')
plt.show()
