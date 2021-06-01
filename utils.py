from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import plotly.express as px
import pandas as pd
import os


def create_word_embeddings(corpus, vector_size=300, embeddings_path="default.vec"):
    """
    Creates word embeddings with Word2Vec

    :param corpus: list of lists of words
    :type corpus: list of list
    :param vector_size: dim of vectors, defaults to 300
    :type vector_size: integer
    :param embeddings_path: path for saving word embeddings, defaults to 'default.vec'
    :type embeddings_path: string
    :return: instance of Word2Vec
    :rtype: Word2Vec
    """
    if not os.path.exists(embeddings_path):
        word_embedding = Word2Vec(sentences=corpus, vector_size=vector_size,
                                  window=10,  min_count=1, workers=10)
        word_embedding.wv.save_word2vec_format(embeddings_path)


def plot_word_embeddings(word_vectors):
    """
    Creates TSNE values from word embeddings and plots it

    :param word_vectors: word embeddings
    :type word_vectors: instance of Word2Vec
    """
    labels = []
    tokens = []

    for index, word in enumerate(list(word_vectors.items.keys())):
        tokens.append(word_vectors.vectors[index])
        labels.append(word)

    tsne_model = TSNE(perplexity=40, n_components=3, init='pca', n_iter=2500, random_state=23)

    projections = tsne_model.fit_transform(tokens, )

    fig = px.scatter_3d(
        projections, x=0, y=1, z=2,
        color=pd.Series(labels), labels={'color': 'words'}
    )
    fig.update_traces(marker_size=8)
    fig.show()
