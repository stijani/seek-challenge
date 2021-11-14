import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from constants import *


def ngram_vectorize(train_texts, train_labels, val_texts, test_texts):
    """Vectorizes texts as n-gram vectors.

    1 text = 1 tf-idf vector the length of vocabulary of unigrams + bigrams.

    # Arguments
        train_texts: list, training text strings.
        train_labels: np.ndarray, training labels.
        val_texts: list, validation text strings.

    # Returns
        x_train, x_val: vectorized training and validation texts
    """
    # Create keyword arguments to pass to the 'tf-idf' vectorizer.
    kwargs = {
            'ngram_range': NGRAM_RANGE,  # Use 1-grams + 2-grams.
            'dtype': 'int32',
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': TOKEN_MODE,  # Split text into word tokens.
            'min_df': MIN_DOCUMENT_FREQUENCY,
    }
    vectorizer = TfidfVectorizer(**kwargs)

    # Learn vocabulary from training texts and vectorize training texts.
    x_train = vectorizer.fit_transform(train_texts)

    # Vectorize validation texts.
    x_val = vectorizer.transform(val_texts)
    x_test = vectorizer.transform(test_texts)

    # Select top 'k' of the vectorized features.
    selector = SelectKBest(f_classif, k=min(TOP_K, x_train.shape[1]))
    selector.fit(x_train, train_labels)
    x_train = selector.transform(x_train).astype('float32')
    x_val = selector.transform(x_val).astype('float32')
    x_test = selector.transform(x_test).astype('float32')
    return x_train, x_val, x_test


def vectorize_data(train_path, val_path, test_path, text_col='description', label_col='target'):
    """
    Loads 3 splits of data frames, vectorizes the the specified text column
    and returns the a vector of features, a vector of labels for each data split.
    
    :param
    train_path (str) - path to the train frame
    val_path (str) - path to the val frame
    test_path (str) - path to the test frame
    text_col (str) - targeted column for vectorization
    label_col (str) - label column in the dataframes
    
    :return
    final_train_vector (np.ndarray) - output train vector
    final_val_vector (np.ndarray) - output val vector
    final_test_vector (np.ndarray) - output test vector
    train_labels (np.ndarray) - output train labels vector
    val_labels (np.ndarray) - output val labels vector
    test_labels (np.ndarray) - output test labels vector
        
    """
    train_df = pd.read_csv(train_path) 
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    train_df = train_df[train_df[text_col].notna()]
    val_df = val_df[val_df[text_col].notna()]
    test_df = test_df[test_df[text_col].notna()]

    # text features
    train_texts = list(train_df[text_col])
    val_texts = list(val_df[text_col])
    test_texts = list(test_df[text_col])

    # labels
    train_labels = list(train_df[label_col])
    val_labels = list(val_df[label_col])
    test_labels = list(test_df[label_col])

    # text feature vectorized and converted to arrays
    train_vector, val_vector, test_vector = ngram_vectorize(train_texts, train_labels, val_texts, test_texts)
    train_vector, val_vector, test_vector = train_vector.toarray(), val_vector.toarray(), test_vector.toarray()

    # numerical features
    train_numerical = train_df.drop([text_col, label_col], axis=1).to_numpy()
    val_numerical = val_df.drop([text_col, label_col], axis=1).to_numpy()
    test_numerical = test_df.drop([text_col, label_col], axis=1).to_numpy()

    # final train and val vectors
    final_train_vector = np.concatenate((train_vector, train_numerical), axis=1)
    final_val_vector = np.concatenate((val_vector, val_numerical), axis=1)
    final_test_vector = np.concatenate((test_vector, test_numerical), axis=1)

    #labels as numpy arrays
    train_labels = np.array(train_df[label_col])
    val_labels = np.array(val_df[label_col])
    test_labels = np.array(test_df[label_col])
    
    return [final_train_vector, 
            final_val_vector, 
            final_test_vector, 
            train_labels, 
            val_labels, 
            test_labels]