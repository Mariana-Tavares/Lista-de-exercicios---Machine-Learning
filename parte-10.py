import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.ensemble
import sklearn.metrics
import sklearn.tree

dfCancer = pd.read_csv('cancer-dados.csv')

dfCancer = dfCancer.drop(['Unnamed: 32'], axis=1)

dfCancer['diagnosis'] = dfCancer['diagnosis'].map({'M': 1, 'B': 0})

X = dfCancer.drop(['id', 'diagnosis'], axis=1)
y = dfCancer['diagnosis']

xTreino, xTeste, yTreino, yTeste = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

randomForest = sklearn.ensemble.RandomForestClassifier(n_estimators=100, random_state=42)
randomForest.fit(xTreino, yTreino)

ypredRandomForest = randomForest.predict(xTeste)

acuracia = sklearn.metrics.accuracy_score(yTeste, ypredRandomForest)
print(f"Acurácia modelo Random Forest: {acuracia:.2f}")

importancias = randomForest.feature_importances_
indicesImportancia = np.argsort(importancias)[::-1]

print("Importância das características:")
for idx in indicesImportancia:
    print(f"{X.columns[idx]}: {importancias[idx]:.4f}")

plt.figure(figsize=(10,6))
plt.title("Importância das características - Random Forest")
plt.barh(X.columns[indicesImportancia], importancias[indicesImportancia], align='center')
plt.xlabel("Importância relativa")
plt.gca().invert_yaxis()
plt.show()

arvoreIndividual = randomForest.estimators_[0]

arvoreTexto = sklearn.tree.export_text(arvoreIndividual, feature_names=list(X.columns))
print("\nÁrvore de decisão individual:")
print(arvoreTexto)

plt.figure(figsize=(20,10))
sklearn.tree.plot_tree(arvoreIndividual, feature_names=X.columns, filled=True, rounded=True, class_names=['Benigno', 'Maligno'])
plt.title("Árvore de decisão individual")
plt.show()
