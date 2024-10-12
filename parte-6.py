import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

dfProdutos = pd.read_csv('data.csv', encoding='ISO-8859-1')

dfProdutos['Description'] = dfProdutos['Description'].fillna('')

dfProdutos = dfProdutos.drop_duplicates(subset='Description')

tfidf = TfidfVectorizer(stop_words='english')
matrizTfidf = tfidf.fit_transform(dfProdutos['Description'])

simCos = cosine_similarity(matrizTfidf, matrizTfidf)

def recomendacao(descricaoProduto, simCos=simCos):
    try:
        idx = dfProdutos[dfProdutos['Description'] == descricaoProduto].index[0]
    except IndexError:
        print("Produto não encontrado.")
        return []

    pontuacaoSimilares = list(enumerate(simCos[idx]))

    pontuacaoSimilares = sorted(pontuacaoSimilares, key=lambda x: x[1], reverse=True)
    pontuacaoSimilares = [x for x in pontuacaoSimilares if x[0] != idx][:5]

    indicesProduto = [i[0] for i in pontuacaoSimilares]

    return dfProdutos['Description'].iloc[indicesProduto]

# Recomenda produtos semelhantes a 'EDWARDIAN PARASOL BLACK'
descricaoProduto = 'EDWARDIAN PARASOL BLACK'
produtosRecomendados = recomendacao(descricaoProduto)

print(f"Produtos recomendados baseados em '{descricaoProduto}':")
print(produtosRecomendados)
