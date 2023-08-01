import gensim
import gensim.downloader as api
from gensim.models.word2vec import Word2Vec

corpus = api.load('text8')
print("Start Training")
model = Word2Vec(corpus)
model.save("trainedModel.bin")
