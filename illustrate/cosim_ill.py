from gensim.models import Word2Vec

model = Word2Vec.load("trainedModel.bin")

print(
    f"Similarity between france and spain: {model.wv.similarity('france', 'spain')}")
print(
    f"Similarity between bsketball and pluto: {model.wv.similarity('basketball', 'pluto')}")
