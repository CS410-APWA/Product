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
import xlrd
import re


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
    sheet = workbook.sheet_by_index(0)

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
            adjusted_score = 10 - int(sheet.cell_value(i, 2)) + 1

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

    tokenized_essays = {}
    for (filename, corpus) in essays.items():
        tokenized_essays[filename] = gensim.utils.simple_preprocess(
            corpus, deacc=True, min_len=2, max_len=20)

    return tokenized_essays


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
    lemmatizer.lemmatize(word, get_wordnet_pos(word))

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
                stopwords_free_essays[label].append(word)

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
    documents = []
    for i, doc in enumerate(preprocessed_essays.values()):
        documents.append(TaggedDocument(doc, [i]))

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

    cluster_df = standardized_df
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


def get_filenames_per_cluster(cluster_df, num_of_clusters):
    """
    Gets all filenames within each of the clusters.

    Args:
        cluster_df : A DataFrame of essays w/ corresponding cluster number.
        num_of_clusters : Predetermined number of cluster (int).

    Returns:
        A dictionary of cluster_id (int) -> filename (string)).
    """

    filenames_per_cluster = {}

    for i in range(num_of_clusters):
        filenames_per_cluster[i] = list(
                                cluster_df[cluster_df.cluster == i].filename)

    return filenames_per_cluster


def update_df_with_topic_scores(cluster_df, num_of_clusters, topic_term_dict):
    """
    Updates the Dataframe containing essays w/ clusters, w/ scores corresponding
    to the topics.

    Args:
        cluster_df : A DataFrame of essays w/ corresponding cluster number.
        num_of_clusters : Predetermined number of cluster (int).
        topic_term_dict : Dictionary of topics w/ defining terms.

    Returns:
        An updated version of the given dataframe (cluster_df).
    """

    # Add cloumn for each topic, and initialize all essay scores to 0.
    for topic in topic_term_dict:
        cluster_df[topic] = 0

    filenames_per_cluster = get_filenames_per_cluster(cluster_df,
                                                      num_of_clusters)
    for i in range(num_of_clusters):
        for filename in filenames_per_cluster[i]:
            essay = list(cluster_df[cluster_df.filename == filename].essay)
            dictionary = corpora.Dictionary(essay)
            essay_corpus = [dictionary.doc2bow(token) for token in essay]
            lda = models.ldamodel.LdaModel(corpus=essay_corpus,
                                           id2word=dictionary,
                                           num_topics=1, passes=10)

            # Get"topic terms" for each essay using LDA.
            essay_term_score = {}
            for idx, terms in lda.print_topics(0, 100):

                # LDA generates topic terms in the format: "term1*score1 + term2*score2 + ..."".
                for term_with_score in terms.split('+'):

                    # Separate terms/scores from LDA generated string.
                    term = term_with_score.split('*')[1][1:-2]
                    score = term_with_score.split('*')[0]

                    # Build a dictionary of all the topic terms of the essay w/ corresponding scores.
                    essay_term_score[term] = float(score)

            # For each topic term extracted for an essay, check if it's in the topic dictionary.
            essay_topic_term_score = {}
            for term in essay_term_score.keys():
                for topic in topic_term_dict.keys():
                    if term in topic_term_dict[topic].keys():
                        # If a term is found in the the topic dictionary, compute it's score and update its value.
                        score = essay_term_score[term] * topic_term_dict[topic][term]
                        if topic in essay_term_score:
                            essay_topic_term_score[topic] += score
                        else:
                            essay_topic_term_score[topic] = score

            # For each essay, add a score corresponding to a topic that corresponds to it.
            for topic, score in essay_topic_term_score.items():
                cluster_df.loc[cluster_df[cluster_df['filename'] == filename].index, topic] = score

    return cluster_df


def update_essay_rank(cluster_df, topic_term_dict, num_clusters):
    """
    Ranks the essays by ... (Matt?)

    Args:

    Returns:

    """
    def distance(v1, v2):
        # L2 norm
        return np.linalg.norm(v1-v2)

    # calculate global centroids for each topic
    topic_globcentroid = {}

    for topic in topic_term_dict.keys():
        # get all vector columns matching topic
        sub_df = cluster_df[cluster_df[topic] != 0].iloc[:, :100]

        # number of vectors with that topic
        n = len(sub_df)

        # mean of all vectors is centroid
        globalcentroid = sum([sub_df.iloc[i] for i in range(n)])/n
        topic_globcentroid[topic] = globalcentroid

    # calculate local centroids for each topic
    for cluster in range(num_clusters):
        for topic in topic_term_dict.keys():
            # get all vector columns matching cluster and topic
            sub_df = cluster_df[cluster_df['cluster'] == cluster]
            sub_df = sub_df[sub_df[topic] != 0].iloc[:, :100]

            # number of vectors in cluster with this topic
            n = len(sub_df)

            # mean of all vectors is centroid with this topic
            localcentroid = sum([sub_df.iloc[i] for i in range(n)])/n

            # find distance between current localcentroid and its corresponding
            # globalcentoid by topic
            d1 = distance(topic_globcentroid[topic], localcentroid)

            # find distance between each vector and its corresponding
            # localcentroid, update rank
            vectors = cluster_df[cluster_df['cluster'] == cluster]
            vectors = vectors[vectors[topic] != 0].index
            for ident in vectors:
                loc = cluster_df.iloc[ident, :100]
                d2 = distance(loc, localcentroid)

                # update rank - lda score * dist from v to localcentroid * dist
                # from localcentroid to globalcentroid
                cluster_df.at[ident, topic] = cluster_df.at[ident, topic] * d1 * d2

    all_we_need = cluster_df.iloc[:, 102:]

    return all_we_need


def main():
    # Upload (from some directory) and store the essays.
    root = os.path.dirname(os.path.realpath('__file__'))
    essay_path = root + '/../essays/'
    essays = upload_essays(essay_path)

    # Read in spreadsheet of topics/defining terms in order to:
    #   1. Build a dictionary of topics w/ defining terms + scores.
    #   2. Preprocess compound defining terms in the essays.
    spreadsheet_path = "topic_term_sheet.xlsx"
    topic_dict, essays = build_dict_of_topics_and_process_compound_terms(
                                essays, spreadsheet_path)

    # Tokenize essays.
    essays = tokenize_essays(essays)


    # Lemmatize all essay tokens.
    essays = lemmatize_essays(essays)

    # Remove any tokens identified as stop words.
    essays = remove_stopwords(essays)

    # Get the vector representation of essays.
    vectorized_essays_df = vectorize_essays(essays)

    # Cluster essays using k-means.
    num_of_clusters = 7     # Should maybe be a global var?

    essays_with_assigned_cluster_df = cluster_with_kmeans(vectorized_essays_df,
                                                          num_of_clusters,
                                                          essays)

    # For each essay, assign an initial score to each of its relevent topics.
    essay_with_topic_scores_df = update_df_with_topic_scores(
                                    essays_with_assigned_cluster_df,
                                    num_of_clusters, topic_dict)

    # Adjust the initially assigned scores.
    all_we_need = update_essay_rank(essay_with_topic_scores_df,
                                    topic_dict, num_of_clusters)


if __name__ == "__main__":
    main()
