from gensim.utils import simple_preprocess
from gensim.models import Word2Vec
from datasets import load_dataset

book_corpus = load_dataset("bookcorpus")
print("Corpus Loaded ")

train_data = book_corpus["train"]
print("Training Data Ready ")

sentences = [simple_preprocess(sentence["text"]) for sentence in train_data]
print("Data Processed ")

model = Word2Vec(sentences, vector_size=100, window=5, min_count=5, workers=4)
print("Vector Embedded")

word_vector = model.wv["word"]

print("Word vector for 'word':", word_vector)

similar_words = model.wv.most_similar("word")
print("Most similar words to 'word':", similar_words)
