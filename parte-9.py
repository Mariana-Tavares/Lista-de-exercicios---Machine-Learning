import numpy as np
import pandas as pd
import sklearn.tree
import sklearn.model_selection
import sklearn.metrics
import sklearn.utils

dfCancer = pd.read_csv('cancer-dados.csv')

dfCancer = dfCancer.drop(['Unnamed: 32'], axis=1)

dfCancer['diagnosis'] = dfCancer['diagnosis'].map({'M': 1, 'B': 0})

X = dfCancer.drop(['id', 'diagnosis'], axis=1)
y = dfCancer['diagnosis']

xTreino, xTeste, yTreino, yTeste = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

arvoreDecisao = sklearn.tree.DecisionTreeClassifier(random_state=42)
arvoreDecisao.fit(xTreino, yTreino)
ypredArvoreDecisao = arvoreDecisao.predict(xTeste)

acuraciaArvore = sklearn.metrics.accuracy_score(yTeste, ypredArvoreDecisao)
print(f"Acurácia árvore de decisão única: {acuraciaArvore:.2f}")

nEstima = 10
predicoesBagging = np.zeros((xTeste.shape[0], nEstima))

for i in range(nEstima):
    xTreinoResample, yTreinoResample = sklearn.utils.resample(xTreino, yTreino, replace=True, random_state=i)
    
    arvoreBagging = sklearn.tree.DecisionTreeClassifier(random_state=i)
    arvoreBagging.fit(xTreinoResample, yTreinoResample)
    
    predicoesBagging[:, i] = arvoreBagging.predict(xTeste)

ypredBagging = np.apply_along_axis(lambda x: np.argmax(np.bincount(x.astype(int))), axis=1, arr=predicoesBagging)

acuraciaBagging = sklearn.metrics.accuracy_score(yTeste, ypredBagging)
print(f"Acurácia modelo Bagging: {acuraciaBagging:.2f}")
