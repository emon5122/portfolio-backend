import ssl

import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    nltk.download("punkt", quiet=True)

stemmer = PorterStemmer()


def tokenize(sentence: str):
    return nltk.word_tokenize(sentence)


def stem(word: str):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence: list, all_words: list):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for index, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[index] = 1.0
    return bag
