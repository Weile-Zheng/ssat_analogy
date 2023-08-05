import numpy as np
# code for cosim can be found in src file folder
from cosim import cosine_similarity
from gensim.models import Word2Vec


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

print()
print("Perimeter is to square as ")
print("a) Chord is to Cylinder")
print("b) Side is to Polygon")
print("c) Degree is to Angle")
print("d) Height is to Pyramid")
print("e) Circumference is to circle")

question = model.wv["perimeter"] - model.wv["square"]
a = model.wv["chord"] - model.wv["cylinder"]
b = model.wv["side"] - model.wv["polygon"]
c = model.wv["degree"] - model.wv["angle"]
d = model.wv["height"] - model.wv["pyramid"]
e = model.wv["circumference"] - model.wv["circle"]

asim = str(cosine_similarity(a, question))
bsim = str(cosine_similarity(b, question))
csim = str(cosine_similarity(c, question))
dsim = str(cosine_similarity(d, question))
esim = str(cosine_similarity(e, question))

print("Cosine Similarities: ")
print("Option A " + asim)
print("Option B " + bsim)
print("Option C " + csim)
print("Option D " + dsim)
print("Option E " + esim)
print()
