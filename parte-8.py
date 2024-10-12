import pandas as pd
import sklearn.model_selection
import sklearn.ensemble
import sklearn.metrics
import tpot

dfCancer = pd.read_csv('cancer-dados.csv')

dfCancer = dfCancer.drop(['Unnamed: 32'], axis=1)

dfCancer['diagnosis'] = dfCancer['diagnosis'].map({'M': 1, 'B': 0})

X = dfCancer.drop(['id', 'diagnosis'], axis=1)
y = dfCancer['diagnosis']

xTreino, xTeste, yTreino, yTeste = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

randomForest = sklearn.ensemble.RandomForestClassifier(random_state=42)
randomForest.fit(xTreino, yTreino)
ypredRandomForest = randomForest.predict(xTeste)
acuraciaManual = sklearn.metrics.accuracy_score(yTeste, ypredRandomForest)
print(f"Acurácia do Random Forest manual: {acuraciaManual:.2f}")

modeloTpot = tpot.TPOTClassifier(
    verbosity=2,
    generations=5,
    population_size=30,
    config_dict='TPOT sparse',
    random_state=42,
    n_jobs=-1
)
modeloTpot.fit(xTreino, yTreino)

ypredTpot = modeloTpot.predict(pd.DataFrame(xTeste, columns=X.columns))

acuraciaTpot = sklearn.metrics.accuracy_score(yTeste, ypredTpot)
print(f"Acurácia do TPOT: {acuraciaTpot:.2f}")

print("\nMelhor pipeline encontrado do TPOT:")
print(modeloTpot.fitted_pipeline_)

modeloTpot.export('tpotMelhorPipeline.py')
