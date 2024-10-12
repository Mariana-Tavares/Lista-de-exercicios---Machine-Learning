import pandas as pd
import sklearn.model_selection
import sklearn.ensemble
import sklearn.tree
import sklearn.metrics

dfCancer = pd.read_csv('cancer-dados.csv')

dfCancer = dfCancer.drop(['Unnamed: 32'], axis=1)

dfCancer['diagnosis'] = dfCancer['diagnosis'].map({'M': 1, 'B': 0})

X = dfCancer.drop(['id', 'diagnosis'], axis=1)
y = dfCancer['diagnosis']

xTreino, xTeste, yTreino, yTeste = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

modeloArvore = sklearn.tree.DecisionTreeClassifier(random_state=42)
modeloArvore.fit(xTreino, yTreino)
ypredArvore = modeloArvore.predict(xTeste)
acuraciaArvore = sklearn.metrics.accuracy_score(yTeste, ypredArvore)
print(f"Acurácia árvore de decisão simples: {acuraciaArvore:.2f}")

modeloAdaBoost = sklearn.ensemble.AdaBoostClassifier(estimator=modeloArvore, random_state=42, n_estimators=100, algorithm='SAMME')
modeloAdaBoost.fit(xTreino, yTreino)
ypredAdaBoost = modeloAdaBoost.predict(xTeste)
acuraciaAdaBoost = sklearn.metrics.accuracy_score(yTeste, ypredAdaBoost)
print(f"Acurácia modelo AdaBoost: {acuraciaAdaBoost:.2f}")

modeloGradientBoosting = sklearn.ensemble.GradientBoostingClassifier(random_state=42, n_estimators=100)
modeloGradientBoosting.fit(xTreino, yTreino)
ypredGradientBoosting = modeloGradientBoosting.predict(xTeste)
acuraciaGradientBoosting = sklearn.metrics.accuracy_score(yTeste, ypredGradientBoosting)
print(f"Acurácia modelo gradient boosting: {acuraciaGradientBoosting:.2f}")
