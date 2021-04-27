import warnings

warnings.filterwarnings(action = 'ignore')

from gensim.models import KeyedVectors, Word2Vec
from multiprocessing import cpu_count
from nltk.tokenize import sent_tokenize, word_tokenize
from pathlib import Path

# Load the text corpus (Harry Potter and The Goblet of Fire).
corpus = Path("the-goblet-of-fire.txt").read_text()
corpus = corpus.replace("\n", " ")

tokenized_corpus = []

# Iterate through tokenized sentences.
for i in sent_tokenize(corpus):

    temp = []

    # Iterate through tokenized words, then append the lowercase version to the training data.
    for j in word_tokenize(i):

        temp.append(j.lower())

    tokenized_corpus.append(temp)

# Create CBOW model and train for 10 epochs.
model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=7, min_count=3, workers=cpu_count(), epochs=30)
# model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=7, min_count=3, workers=cpu_count(), epochs=30, sg=1)

# Save the state of the trained model (can resume additional training later if desired).
model.save("word2vec-cbow.model")

# Save the word vectors (words mapped to their associated word embeddings).
word_vectors = model.wv
word_vectors.save("word2vec-cbow.wordvectors")

# Load the word vectors with memory-mapping as read-only, shared across processes.
wv = KeyedVectors.load("word2vec-cbow.wordvectors", mmap='r')

# Print one of the word embeddings.
print("Word Embedding for \"Harry\"")
print(wv['harry'])
print()

# Print most similar word using the default "cosine similarity" measure.
print("Most Similar Word for \"Harry\" + \"Broomstick\"")
result = wv.most_similar(positive=["harry", "broomstick"])
most_similar_key, similarity = result[0] # Take the first match.
print(f"{most_similar_key}: {similarity:.4f}")
print()

# Print the most similar by word.
print("Most Similar Word for \"Hogwarts\"")
result = word_vectors.similar_by_word("hogwarts")
most_similar_key, similarity = result[0] # Take the first match.
print(f"{most_similar_key}: {similarity:.4f}")
print()

# Print the word that doesn't match.
print("Word That Doesn't Match in [\"Harry\", \"Ron\", \"Hermione\", \"Hogwarts\"]")
print(wv.doesnt_match("harry ron hermione hogwarts".split()))
