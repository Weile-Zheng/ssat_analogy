import numpy as np
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Replace with the path to your trained model file
model = Word2Vec.load("trainedModel.bin")

question = model.wv["flake"] - model.wv["snow"]
a = model.wv["storm"] - model.wv["hail"]
b = model.wv["drop"] - model.wv["rain"]
c = model.wv["field"] - model.wv["wheat"]
d = model.wv["stack"] - model.wv["hay"]
e = model.wv["cloud"] - model.wv["fog"]
vectors = [question, a, b, c, d, e]

# Apply PCA to reduce the word vectors to 2 dimensions
pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(vectors)

# Plot the 2D word vectors using Matplotlib
plt.figure(figsize=(10, 8))
plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], marker='o', s=30)

# Annotate the points with the labels 'question', 'a' to 'e'
labels = ['question', 'a', 'b', 'c', 'd', 'e']
for i, label in enumerate(labels):
    if label == 'question':
        plt.annotate(label, xy=(
            vectors_2d[i, 0], vectors_2d[i, 1]), fontsize=12, color='red')
    else:
        plt.annotate(label, xy=(
            vectors_2d[i, 0], vectors_2d[i, 1]), fontsize=12)

plt.title('Embeddings Visualization')
plt.grid(True)
plt.show()
