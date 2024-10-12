import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.ensemble
import sklearn.tree
import sklearn.metrics

df = pd.read_csv('cancer-dados.csv')

df = df.drop(['Unnamed: 32'], axis=1, errors='ignore')

df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

X = df.drop(['id', 'diagnosis'], axis=1)
y = df['diagnosis']

xTreino, xTeste, yTreino, yTeste = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

modeloBagging = sklearn.ensemble.BaggingClassifier(estimator=sklearn.tree.DecisionTreeClassifier(), n_estimators=100, random_state=42)
modeloBoosting = sklearn.ensemble.AdaBoostClassifier(estimator=sklearn.tree.DecisionTreeClassifier(), n_estimators=100, random_state=42)
modeloRandomForest = sklearn.ensemble.RandomForestClassifier(n_estimators=100, random_state=42)

modeloBagging.fit(xTreino, yTreino)
modeloBoosting.fit(xTreino, yTreino)
modeloRandomForest.fit(xTreino, yTreino)

ypredBagging = modeloBagging.predict(xTeste)
ypredBoosting = modeloBoosting.predict(xTeste)
ypredRf = modeloRandomForest.predict(xTeste)

accBagging = sklearn.metrics.accuracy_score(yTeste, ypredBagging)
accBoosting = sklearn.metrics.accuracy_score(yTeste, ypredBoosting)
accRf = sklearn.metrics.accuracy_score(yTeste, ypredRf)

print(f"Acurácia Bagging: {accBagging:.2f}")
print(f"Acurácia Boosting: {accBoosting:.2f}")
print(f"Acurácia Random Forest: {accRf:.2f}")

def plotarCurvaAprendizado(modelo, titulo, X, y):
    trainSizes, trainScores, testScores = sklearn.model_selection.learning_curve(modelo, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5))

    trainMean = np.mean(trainScores, axis=1)
    testMean = np.mean(testScores, axis=1)
    
    plt.plot(trainSizes, trainMean, label='Acurácia Treino')
    plt.plot(trainSizes, testMean, label='Acurácia Validação')
    plt.title(titulo)
    plt.xlabel('Tamanho do Treinamento')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.grid()
    plt.show()

plotarCurvaAprendizado(modeloBagging, 'Curva de Aprendizado - Bagging', X, y)
plotarCurvaAprendizado(modeloBoosting, 'Curva de Aprendizado - Boosting', X, y)
plotarCurvaAprendizado(modeloRandomForest, 'Curva de Aprendizado - Random Forest', X, y)

print("Análise Comparativa:")
print(f"Bagging: {accBagging:.2f}")
print(f"Boosting: {accBoosting:.2f}")
print(f"Random Forest: {accRf:.2f}")
