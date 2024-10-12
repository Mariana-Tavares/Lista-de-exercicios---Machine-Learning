import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

arquivo = pd.read_csv('cancer-dados.csv')

if 'Unnamed: 32' in arquivo.columns:
    arquivo = arquivo.drop(['Unnamed: 32'], axis=1)

arquivo['diagnosis'] = arquivo['diagnosis'].map({'M': 1, 'B': 0})

colunaNum = arquivo.drop(['id', 'diagnosis'], axis=1).columns
caracSeparado = arquivo[colunaNum].values

xMedia = np.mean(caracSeparado, axis=0)
xDesvioPadrao = np.std(caracSeparado, axis=0)
xPadronizado = (caracSeparado - xMedia) / xDesvioPadrao

covMatriz = np.cov(xPadronizado, rowvar=False)

autovalores, autovetores = np.linalg.eigh(covMatriz)

indiceOrdenado = np.argsort(autovalores)[::-1]
autovaloresOrdenados = autovalores[indiceOrdenado]
autovetoresOrdenados = autovetores[:, indiceOrdenado]

nComp = 2
subconjVetProp = autovetoresOrdenados[:, :nComp]

xReduzido = np.dot(xPadronizado, subconjVetProp)

plt.figure(figsize=(8,6))
plt.scatter(arquivo['radius_mean'], arquivo['texture_mean'], c=arquivo['diagnosis'], cmap='coolwarm')
plt.xlabel('Média do Raio')
plt.ylabel('Média da Textura')
plt.title('Dados Originais: Média do Raio vs Média da Textura')
plt.colorbar(label='Diagnóstico (0=Benigno, 1=Maligno)')
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(xReduzido[:, 0], xReduzido[:, 1], c=arquivo['diagnosis'], cmap='coolwarm')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Dados pós PCA - Reduzidos para 2 Dimensões')
plt.colorbar(label='Diagnóstico (0=Benigno, 1=Maligno)')
plt.show()
