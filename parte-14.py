import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.discriminant_analysis
import sklearn.preprocessing

df = pd.read_csv('cancer-dados.csv')

df = df.drop(['Unnamed: 32'], axis=1)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

X = df.drop(['id', 'diagnosis'], axis=1)
y = df['diagnosis']

escalador = sklearn.preprocessing.StandardScaler()
xPadronizado = escalador.fit_transform(X)

lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(n_components=1)
xLda = lda.fit_transform(xPadronizado, y)

plt.figure(figsize=(8,6))
plt.scatter(xLda[:, 0], np.zeros_like(xLda), c=y, cmap='coolwarm', alpha=0.7)
plt.xlabel('LDA Componente 1')
plt.title('LDA - Separação das Classes')
plt.colorbar(label='Diagnóstico (0=Benigno, 1=Maligno)')
plt.show()

ldaCoefs = lda.scalings_ 
variaveisInfluentes = pd.DataFrame(data=ldaCoefs, index=X.columns, columns=['LDA Componente 1'])
variaveisInfluentes['Abs(Coef)'] = np.abs(variaveisInfluentes['LDA Componente 1'])
variaveisInfluentes = variaveisInfluentes.sort_values(by='Abs(Coef)', ascending=False)

print("Variáveis que mais influenciam na discriminação das classes:")
print(variaveisInfluentes.head(10))
