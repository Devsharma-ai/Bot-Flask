import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk
from nltk.tree import Tree

# Download required datasets
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def tokenize(sentence):
    """
    Split sentence into array of words/tokens.
    """
    return word_tokenize(sentence)


def remove_stopwords(words):
    """
    Remove stop words from tokenized words.
    """
    return [word for word in words if word.lower() not in stop_words]


def pos_tagging(words):
    """
    Assign POS tags to words.
    """
    return pos_tag(words)


def named_entity_recognition(sentence):
    """
    Extract named entities like names, locations, organizations.
    """
    tree = ne_chunk(pos_tag(word_tokenize(sentence)))
    entities = []
    for subtree in tree:
        if isinstance(subtree, Tree):
            entity_name = " ".join([token for token, pos in subtree.leaves()])
            entity_type = subtree.label()
            entities.append((entity_name, entity_type))
    return entities


def stem(word):
    """
    Stemming: find the root form of the word.
    """
    return stemmer.stem(word.lower())


def lemmatize(word):
    """
    Lemmatization: find the base meaningful form of the word.
    """
    return lemmatizer.lemmatize(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
    Create a bag of words array.
    """
    # Stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # Initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1

    return bag
