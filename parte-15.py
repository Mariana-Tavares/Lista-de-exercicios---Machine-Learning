import pandas as pd
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.decomposition
import sklearn.ensemble
import sklearn.pipeline
import sklearn.metrics

df = pd.read_csv('cancer-dados.csv')

df = df.drop(['Unnamed: 32'], axis=1)

df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

X = df.drop(['id', 'diagnosis'], axis=1)
y = df['diagnosis']

xTreino, xTeste, yTreino, yTeste = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = sklearn.pipeline.Pipeline([
    ('escalador', sklearn.preprocessing.StandardScaler()),
    ('pca', sklearn.decomposition.PCA(n_components=2)), 
    ('classificador', sklearn.ensemble.RandomForestClassifier(random_state=42))
])

scores = sklearn.model_selection.cross_val_score(pipeline, xTreino, yTreino, cv=5, scoring='accuracy')

print(f"Desempenho do Pipeline com validação cruzada: {scores}")
print(f"Acurácia média do Pipeline: {scores.mean():.2f}")

pipeline.fit(xTreino, yTreino)
yPred = pipeline.predict(xTeste)

acuracia = sklearn.metrics.accuracy_score(yTeste, yPred)
print(f"Acurácia do Pipeline no conjunto de teste: {acuracia:.2f}")
