import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import wordnet as wn
from bs4 import BeautifulSoup as bs
import spacy
spacy.load('en')
from spacy.lang.en import English
from collections import Counter
import nltk
sns.set_theme()
parser = English()

from constants import *


def strip_html_tags(cell_vlaue, delimeter=' '):
    # TODO: write docstring
    if type(cell_vlaue) == str:
        soup = bs(cell_vlaue)
        text = soup.get_text(delimeter)
        return text
    return cell_vlaue


def tokenize(cell_value):
    # TODO: write docstring
    if type(cell_value) == str:
        processed_tokens = []
        tokens = parser(cell_value)
        for token in tokens:
            if token.orth_.isspace():
                continue
            elif not str(token).isalnum():
                continue
            elif token.like_url:
                processed_tokens.append('URL')
            elif token.orth_.startswith('@'):
                processed_tokens.append('SCREEN_NAME')
            else:
                processed_tokens.append(token.lower_)
        #return ' '.join(processed_tokens)
        return processed_tokens
    return cell_value


def lemmatize(doc):
    # TODO: write docstring
    res = []
    if type(doc) == list:
        for token in doc:
            lemma = wn.morphy(token)
            if lemma is None:
                res.append(token)
            else:
                res.append(lemma)
        return res
    return doc

def remove_stop_words(doc):
    # TODO: write docstring
    res = []
    stop_words = set(nltk.corpus.stopwords.words('english')) 
    if type(doc) == list:
        for token in doc:
            if token in stop_words:
                continue
            res.append(token)
        return ' '.join(res)
    return doc


def data_prep_pipeline(df):
    # TODO: write docstring
    processed_df = df \
            .applymap(strip_html_tags) \
            .applymap(tokenize) \
            .applymap(lemmatize) \
            .applymap(remove_stop_words)
    return processed_df


def check_missing_distribution(df, missing_column, target):
    # TODO: write docstring
    df_miss = df[df[missing_column] == 'NA']
    counts = dict(Counter(df[target]))
    categories, counts = zip(*((cat, num) for cat, num in counts.items()))
    return categories, counts




plt.rc('font', **FONT)

def plot_bar(categories, counts, title, x_label, y_label, rotation=45):
    # TODO: write docstring
    fig, ax = plt.subplots(figsize=PLOT_SIZE)
    ax.bar(categories, counts)
    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=rotation, horizontalalignment='center')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
    

def plot_top_words(df, category, top_n=50):
    # TODO: write docstring
    job_descriptions = df[df['target']==category]['description']
    job_descriptions_merged = ' '.join(job_descriptions).split(' ')
    word_counts = dict(Counter(job_descriptions_merged).most_common())
    word, count = zip(*[(word, count) for word, count in word_counts.items()])

    title = 'Top 50 words in {} Job Descriptions'.format(category)
    x_label = 'Word'
    y_label = 'Count'
    plot_bar(word[:top_n], count[:top_n], title, x_label, y_label, rotation=90) 
    
    
def plot_metrics(history, metric1, metric2, loc='upper right'):
    # TODO: write docstring
    epochs_ = [i for i in range(len(history[metric1]))]
    plt.rc('font', **FONT)
    plt.plot(epochs_, history[metric1], label=metric1)
    plt.plot(epochs_, history[metric2], label=metric2)

    plt.xlabel('Epoch Number')
    plt.ylabel('Metric Values')
    plt.legend(loc=loc)
    plt.show()
            
def plot_conf_matrix(conf_matrix):
    # TODO: write docstring
    plt.figure(figsize=(15,10))
    sns.set(font_scale=1.4) # for label size
    sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 16})
    plt.show()
    
    



