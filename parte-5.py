import pandas as pd
import numpy as np
import sklearn.metrics
import sklearn.model_selection

dados = pd.read_csv('ratings.csv')

matrizAvaliacoes = dados.pivot_table(index='userId', columns='movieId', values='rating')

mediaAvaliacoes = matrizAvaliacoes.mean(axis=1)
avaliacoesSemMedia = matrizAvaliacoes.sub(mediaAvaliacoes, axis=0)

similaridadeUsuario = 1 - sklearn.metrics.pairwise.pairwise_distances(avaliacoesSemMedia.fillna(0), metric="correlation")

similaridadeUsuarioDf = pd.DataFrame(similaridadeUsuario, index=matrizAvaliacoes.index, columns=matrizAvaliacoes.index)

def preverAvaliacoesUsuario(avaliacoes, similaridade):
    mediaAvaliacaoUsuario = avaliacoes.mean(axis=1).values
    diferencaAvaliacoes = (avaliacoes.values - mediaAvaliacaoUsuario[:, np.newaxis])
    previsao = mediaAvaliacaoUsuario[:, np.newaxis] + similaridade.dot(np.nan_to_num(diferencaAvaliacoes)) / np.array([np.abs(similaridade).sum(axis=1)]).T
    return previsao

previsoes_usuario_pearson = preverAvaliacoesUsuario(matrizAvaliacoes, similaridadeUsuarioDf)

similaridade_item_pearson = 1 - sklearn.metrics.pairwise.pairwise_distances(matrizAvaliacoes.T.fillna(0), metric="correlation")

def preverAvaliacoesItem(avaliacoes, similaridade):
    previsao = avaliacoes.dot(similaridade) / np.array([np.abs(similaridade).sum(axis=1)])
    return previsao

previsoesItemPearson = preverAvaliacoesItem(matrizAvaliacoes, similaridade_item_pearson)

dadoTreino, dadosTeste = sklearn.model_selection.train_test_split(dados, test_size=0.2)

mTeste = dadosTeste.pivot_table(index='userId', columns='movieId', values='rating')

usuarioTeste = mTeste.index
itemTeste = mTeste.columns

previsaoUsuarioDf = pd.DataFrame(previsoes_usuario_pearson, index=matrizAvaliacoes.index, columns=matrizAvaliacoes.columns)
previsaoItemDf = pd.DataFrame(previsoesItemPearson, index=matrizAvaliacoes.index, columns=matrizAvaliacoes.columns)

previsaoTesteUsuario = previsaoUsuarioDf.loc[usuarioTeste, itemTeste]
previsaoTesteItem = previsaoItemDf.loc[usuarioTeste, itemTeste]

mseUsuario = sklearn.metrics.mean_squared_error(mTeste.fillna(0).values.flatten(), previsaoTesteUsuario.fillna(0).values.flatten())
sisUsuario = np.sqrt(mseUsuario)

mseItem = sklearn.metrics.mean_squared_error(mTeste.fillna(0).values.flatten(), previsaoTesteItem.fillna(0).values.flatten())
sisItem = np.sqrt(mseItem)

print(f"Sistema de recomendação baseado em filtragem colaborativa baseada em usuário: {sisUsuario:.2f}")
print(f"Sistema de recomendação baseado em filtragem colaborativa baseada em item: {sisItem:.2f}")
