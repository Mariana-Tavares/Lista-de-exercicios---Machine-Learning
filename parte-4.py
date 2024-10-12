import pandas as pd
import sklearn.model_selection
import sklearn.discriminant_analysis
import sklearn.preprocessing
import sklearn.metrics

arquivo = pd.read_csv('cancer-dados.csv')

if 'Unnamed: 32' in arquivo.columns:
    arquivo = arquivo.drop(columns=['Unnamed: 32'])

arquivo['diagnosis'] = arquivo['diagnosis'].map({'M': 1, 'B': 0})

caracSeparado = arquivo.drop(columns=['id', 'diagnosis'])
caracSeparado2 = arquivo['diagnosis']

xTreino, xTeste, yTreino, yTeste = sklearn.model_selection.train_test_split(
    caracSeparado, caracSeparado2, test_size=0.3, random_state=42)

escalador = sklearn.preprocessing.StandardScaler()
xTreinoPadronizado = escalador.fit_transform(xTreino)
xTestePadronizado = escalador.transform(xTeste)

lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
lda.fit(xTreinoPadronizado, yTreino)

yPrevisto = lda.predict(xTestePadronizado)

precisao = sklearn.metrics.accuracy_score(yTeste, yPrevisto)
print(f"Precisão do modelo LDA: {precisao * 100:.2f}%")
