import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.decomposition
import sklearn.preprocessing

df = pd.read_csv('cancer-dados.csv')

df = df.drop(['Unnamed: 32', 'id'], axis=1)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

escalador = sklearn.preprocessing.StandardScaler()
xPadronizado = escalador.fit_transform(X)

pca = sklearn.decomposition.PCA(n_components=2)
xPca = pca.fit_transform(xPadronizado)

varianciaExplicada = pca.explained_variance_ratio_
print(f"Variância explicada pelo Componente Principal 1: {varianciaExplicada[0]:.2f}")
print(f"Variância explicada pelo Componente Principal 2: {varianciaExplicada[1]:.2f}")

plt.figure(figsize=(8,6))
plt.scatter(xPca[:, 0], xPca[:, 1], c=y, cmap='coolwarm', alpha=0.6)
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('PCA - Reduzido para 2 Dimensões')
plt.colorbar(label='Diagnóstico (0=Benigno, 1=Maligno)')
plt.show()

componentesPca = pd.DataFrame(pca.components_, columns=X.columns, index=['Componente Principal 1', 'Componente Principal 2'])
print("\nContribuição de cada característica nos componentes principais:")
print(componentesPca.T)

plt.figure(figsize=(10,6))
plt.bar(x=componentesPca.columns, height=componentesPca.iloc[0], alpha=0.7, label="Componente Principal 1")
plt.bar(x=componentesPca.columns, height=componentesPca.iloc[1], alpha=0.5, label="Componente Principal 2")
plt.xticks(rotation=90)
plt.title('Contribuição das Características nos Componentes Principais')
plt.ylabel('Peso')
plt.legend()
plt.tight_layout()
plt.show()
