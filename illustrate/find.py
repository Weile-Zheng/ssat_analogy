from gensim.models import Word2Vec

model = Word2Vec.load("trainedModel.bin")
print(model.wv.most_similar('dog'))
