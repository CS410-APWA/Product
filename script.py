import chardet
import csv
import gensim
import nltk
import os
import string
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim import corpora, models


def upload_essays(essay_path):
    """Uploads essays from given path and stores them in a dictionary.

    Args:
        essay_path: A string representing the path to the essays directory.

    Returns:
        A dictionary of filename (string) -> essay corpus (string).

    Raises:
        Error if path to essay directory is not valid.
    """

    # Should raise error if path not valid: maybe use try/except?
    files = os.listdir(essay_path)

    essays = {}
    for file in files:
        # Attempt to confidently guess encoding;
        # Otherwise, default to ISO-8859-1.
        encoding = "ISO-8859-1"
        guess = chardet.detect(open(essay_path + file, "rb").read())

        if (guess["confidence"] >= 0.95):
            encoding = guess["encoding"]

        with open(essay_path + file, "r", encoding=encoding) as f:
            essays[file] = f.read()

    return essays


def build_dict_of_topics_and_process_compound_terms(essays, sheet_path):
    """
        - Reads a spreadsheet of topics/defining terms, and builds a Dictionary of
    topic -> (defining_term -> score).
        - Checks for compound defining terms and processes them in essays.

    Args:
        essays: Dictionary of filename (string) -> essay (string).

    Returns:
        A tuple: Dictionary of topics, Dictionary of essays w/ procecessed
        compound terms.
    """

    workbook = xlrd.open_workbook(sheet_path)
    sheet = wb.sheet_by_index(0)

    theme_term_dict = {}

    # Read the first column (TOPIC) and add topics as keys to the dictionary.
    current_topic = ""
    for i in range(1, sheet.nrows):
        topic = sheet.cell_value(i, 0)
        if topic:
            theme_term_dict[topic] = {}
            current_topic = topic

        term = sheet.cell_value(i, 1)
        if term:
            # Compound ? If yes, remove spacing.
            if ' ' in term:
                spacefree_term = ''.join(term.split(' '))

                # Replace all occurences of compound terms by removing spaces.
                for (label, corpus) in essays.items():
                    re.sub(term, spacefree_term, corpus)

            # Lemmatize.
            lemmatized_term = lemmatize_word(spacefree_term)

            # Append lemmatized terms.
            adjusted_score = 10 - sheet.cell_value(i, 2) + 1

            # Store term + score in dictionary (no duplicates).
            if lemmatized_term not in theme_term_dict[current_topic]:
                theme_term_dict[current_topic][lemmatized_term] = adjusted_score

    return theme_term_dict, essays


def tokenize_essays(essays):
    """
    Converts each essay from a string to a list of strings (tokens), while
    disregarding words that are too short/long.

    Args:
        essays: A dictionary of filename (string) -> essay corpus (string).

    Returns:
        A dicionary of filename (string) -> tokenized corpus (list of strings).
    """

    tokenize_essays = {}
    for (filename, corpus) in essays.items():
        tokenized_essays[filename] = gensim.utils.simple_preprocess(
            corpus, deacc=True, min_len=2, max_len=20)

    return tokenize_essays


def lemmatize_word(word):
    """
    Converts a given word (string) to its lemmatized version.

    Args:
        word (string)

    Returns:
        The lemmatized version of the word (string).
    """

    def get_wordnet_pos(word):
        """Map POS tag to first character lemmatize() accepts.

        Args:
            word (string).

        Returns:
            A part of speech parameter that charactrizes the given word (char).
        """
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": nltk.corpus.wordnet.ADJ,
                    "N": nltk.corpus.wordnet.NOUN,
                    "V": nltk.corpus.wordnet.VERB,
                    "R": nltk.corpus.wordnet.ADV}

        return tag_dict.get(tag, nltk.corpus.wordnet.NOUN)

    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmatizer.lemmatize(word, get_wordnet_pos(word)

    return word


def lemmatize_essays(tokenized_essays):
    """
    Converts the tokens (words) of each essay into lemmatized tokens.

    Args:
        A dicionary of filename (string) -> tokenized corpus (list of strings).

    Returns:
        A dicionary of filename (string) -> tokenized+lemmatized corpus.
    """

    lemmatized_essays = {}
    for (label, word_lst) in tokenized_essays.items():
        lemmatized_essays[label] = []
        for word in word_lst:
            lemmatized_essays[label].append(lemmatize_word(word))

    return lemmatized_essays


def remove_stopwords(lemmatized_essays):
    """
    Removes any tokens charactrized as stop words from the essay tokens.

    Args:
        A dicionary of filename (string) -> tokenized+lemmatized corpus.

    Returns:
        A dicionary of filename (string) -> essay corpus w/o stop words.
    """

    english_stopwords = nltk.corpus.stopwords.words('english')
    custom_stopwords = open("custom_stopwords.txt", "r").read().splitlines()

    stopwords_free_essays = {}
    for (label, word_lst) in lemmatized_essays.items():
        stopwords_free_essays[label] = []
        for word in word_lst:
            if word not in english_stopwords + custom_stopwords:
                stopwords_free_essays.append(word)

    return stopwords_free_essays


def vectorize_essays(preprocessed_essays):
    """
    Converts each essay into a vector representation using Doc2Vec.

    Args:
        A dictionary of tokenized + lemmatized + stopwords_free essays.

    Returns:
        A Dataframe of essays (rows) and vector representation (cols) in
        100 dimensions.
    """

    # Vectorize w/ doc2vec.
    for i, doc in enumerate(preprocessed_essays.values()):
        documents = [TaggedDocument(doc, [i]) ]

    d2v_model = Doc2Vec(documents, vector_size=100)
    vectorized_df = pd.DataFrame(d2v_model.docvecs.vectors_docs)

    # Feature scaling through standardization.
    stdsclr = StandardScaler()
    standardized_df = pd.DataFrame(
            stdsclr.fit_transform(vectorized_df.astype(float)))

    return standardized_df


def cluster_with_kmeans(standardized_df, num_of_clusters, preprocessed_essays):
    """
    Partitions essays into clusters using k-means.

    Args:
        standardized_df: Vector representation of essays (DataFrame).
        num_of_clusters: Predetermined number of cluster (int).
        preprocessed_essays: Dictionary of essays (filename -> corpus tokens).

    Returns:
        A DataFrame of essays w/ corresponding cluster number
        (row: essays, cols: cluster id, essay corpus, filename).
    """

    kmeans = KMeans(n_clusters=num_of_clusters, init="k-means++", max_iter=100)
    kmeans.fit(standardized_df.values)

    cluster_df = pd.DataFrame()
    cluster_df['cluster'] = kmeans.labels_
    cluster_df['essay'] = preprocessed_essays.values()
    cluster_df['filename'] = preprocessed_essays.keys()

    return cluster_df


def get_essays_per_cluster(cluster_df, num_of_clusters):
    """
    Gets all essays corpuses within each of the clusters.

    Args:
        cluster_df : A DataFrame of essays w/ corresponding cluster number.
        num_of_clusters : Predetermined number of cluster (int).

    Returns:
        A dictionary of cluster_id (int) -> essays (list of lists of strings).
    """

    essays_per_cluster = {}

    for i in range(num_clusters):
        essays_per_cluster[i] = list(output[output.cluster == i].essay)

    return essays_per_cluster


def main():
    # Upload (from some directory) and store the essays.
    root = os.path.dirname(os.path.realpath('__file__'))
    essay_path = root + '/../essays/'
    essays = upload_essays(essay_path)

    # Read in spreadsheet of topics/defining terms in order to:
    #   1. Build a dictionary of topics w/ defining terms + scores.
    #   2. Preprocess compound defining terms in the essays.
    spreadsheet_path = "theme_term_dic.xls"
    topic_dict, essays = build_dict_of_topics_and_process_compound_terms(
                                initial_essays, spreadsheet_path)

    # Tokenize essays.
    essays = tokenize_essays(essays)

    # Lemmatize all essay tokens.
    essays = lemmatize_essays(tokenized_essays)

    # Remove any tokens identified as stop words.
    essays = remove_stopwords(lemmatized_essays)

    # Get the vector representation of essays.
    vectorized_essays_df = vectorize_essays(stopwords_free_essays)

    # Cluster essays using k-means.
    num_of_clusters = 7     # Should maybe be a global var?

    clustered_essays_df = cluster_with_kmeans(vectorized_essays, num_of_clusters,
                                              essays)


if __name__ == "__main__":
    main()
