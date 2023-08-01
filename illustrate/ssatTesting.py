from typing import final
from gensim.models import Word2Vec
import numpy as np


model = Word2Vec.load("trainedModel.bin")

paris_vector = model.wv["paris"]
france_vector = model.wv["france"]
england_vector = model.wv["england"]

difference_vector = paris_vector - france_vector
print("The difference between paris and france: " + str(difference_vector))

final_vector = england_vector + difference_vector
final_word, sim = model.wv.similar_by_vector(final_vector, topn=2)[1]

print("A final similar word after adding the difference between paris and france to england: " + final_word)
