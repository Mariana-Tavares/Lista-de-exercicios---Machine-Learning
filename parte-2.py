import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
import sklearn.decomposition

arquivo = pd.read_csv('cancer-dados.csv')

if 'Unnamed: 32' in arquivo.columns:
    arquivo = arquivo.drop(['Unnamed: 32'], axis=1)

arquivo['diagnosis'] = arquivo['diagnosis'].map({'B': 0, 'M': 1})

colunasNumericas = arquivo.drop(['id', 'diagnosis'], axis=1).columns

caracSeparado = arquivo[colunasNumericas]

escalador = sklearn.preprocessing.StandardScaler()
xPadronizado = escalador.fit_transform(caracSeparado)

pca = sklearn.decomposition.PCA(n_components=2)
xPca = pca.fit_transform(xPadronizado)

varianciaExplicada = pca.explained_variance_ratio_
print(f"Variância explicada pelo Componente Principal 1: {varianciaExplicada[0]:.2f}")
print(f"Variância explicada pelo Componente Principal 2: {varianciaExplicada[1]:.2f}")

plt.figure(figsize=(8,6))
plt.scatter(xPca[:, 0], xPca[:, 1], c=arquivo['diagnosis'], cmap='coolwarm')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Dados após Scikit-Learn - Reduzidos para 2 Dimensões')
plt.colorbar(label='Diagnóstico (0=Benigno, 1=Maligno)')
plt.show()
