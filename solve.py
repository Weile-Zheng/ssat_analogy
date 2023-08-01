import numpy as np
from gensim.models import Word2Vec


def cosine_similarity(vector1, vector2):
    # Calculate the cosine distance between the two vectors using NumPy
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))


model = Word2Vec.load("trainedModel.bin")

question = model.wv["flake"] - model.wv["snow"]
a = model.wv["storm"] - model.wv["hail"]
b = model.wv["drop"] - model.wv["rain"]
c = model.wv["field"] - model.wv["wheat"]
d = model.wv["stack"] - model.wv["hay"]
e = model.wv["cloud"] - model.wv["fog"]

print("Cosine Similarities: ")
print("Option A " + str(cosine_similarity(a, question)))
print("Option B " + str(cosine_similarity(b, question)))
print("Option C " + str(cosine_similarity(c, question)))
print("Option D " + str(cosine_similarity(d, question)))
print("Option E " + str(cosine_similarity(e, question)))
