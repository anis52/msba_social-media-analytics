import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
from gensim.models import word2vec, Word2Vec
from sklearn.manifold import TSNE
from sklearn.cluster import AffinityPropagation, KMeans
from sklearn.decomposition import PCA


print('import ok')



def normalize_document(doc):
    doc = re.sub(r'[^a-zA-Z\s]', '', doc)
    doc = doc.lower()
    doc = doc.strip()
    tokens = wpt.tokenize(doc)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    doc = ' '.join(filtered_tokens)
    return doc

print('normalize_document ok')

# average the word vectors in a document (tokenized words)
def avg_word_vectors(words, model, vocabulary, num_features):
    feature_vector = np.zeros((num_features,), dtype="float64")
    nwords = 0.
    for word in words:
        if word in vocabulary:
            nwords = nwords + 1.
            feature_vector = np.add(feature_vector, model[word])  # add the word vector if the word is in the dictionary
    if nwords:
        feature_vector = np.divide(feature_vector, nwords)  # averaging across all words
    return feature_vector

print('avg_word_vectors ok')

# average the word vectors for a corpus
def avg_word_vectorizer(corpus,model,num_features):
    vocabulary=set(model.wv.index2word)
    features=[avg_word_vectors(tokenized_sentence,model,vocabulary,num_features) for tokenized_sentence in corpus]
    return np.array(features)

print('avg_word_vectorizer ok')


#Read csv

# read csv file
df = pd.read_csv("posts.csv", index_col= None)

print(df)

print('import csv ok')

df = df.dropna()


# Group by each film, concatenate the text string
df['posttext'] = df.groupby(['imdb_id'])['posttext'].transform(lambda x: ','.join(x))

# drop duplicate data
df = df.drop_duplicates()
df = df.reset_index(drop=True)
# show the dataframe
print(df)




for i in range(1):


    # text normalization
    wpt = nltk.WordPunctTokenizer()
    stop_words = nltk.corpus.stopwords.words('english')

    normalize_df = np.vectorize(normalize_document)
    norm_df = normalize_df(df.loc[i, 'posttext'].split(','))
    print('### input text data after preprocessing')
    print(norm_df)

    # bag of words
    cv = CountVectorizer(min_df=0.1, max_df=1.)
    cv_matrix = cv.fit_transform(norm_df)
    print('### word counts within each text document')
    print(cv_matrix)

    cv_matrix = cv_matrix.toarray()
    print('### word counts in the array format: one array for one text document')
    print(cv_matrix)

    vocab = cv.get_feature_names_out()
    print('### word counts in a data frame format with columns being the words in the dictionary')
    print(pd.DataFrame(cv_matrix, columns=vocab))

    # bag of n-grams
    # set n-gram range to 1,2 to get both unigrams and bigrams
    bv = CountVectorizer(ngram_range=(1, 2))
    bv_matrix = bv.fit_transform(norm_df)
    print('### unigram and bigram counts within each text document')
    print(bv_matrix)
    bv_matrix = bv_matrix.toarray()
    print('### unigram and bigram counts in the array format: one array for one text document')
    print(bv_matrix)
    vocab = bv.get_feature_names_out()
    print('### unigram and bigram counts in a data frame format with columns being the words in the dictionary')
    print(pd.DataFrame(bv_matrix, columns=vocab))

    # tf-idf
    print('### tf-idf counts')
    print('### Approach 1: CountVectorizer + Tfidftransformer')
    tt = TfidfTransformer(norm='l2', use_idf=True)
    tt_matrix = tt.fit_transform(cv_matrix)
    print('### tf-idf counts within each text document')
    print(tt_matrix)
    tt_matrix = tt_matrix.toarray()
    print('### tf-idf counts in the array format: one array for one text document')
    print(tt_matrix)
    vocab = cv.get_feature_names_out()
    print('### tf-idf counts in a data frame format with columns being the words in the dictionary')
    print(pd.DataFrame(np.round(tt_matrix, 2), columns=vocab))

    print('### Approach 2: TfidfVectorizer')
    tv = TfidfVectorizer(min_df=0., max_df=1., norm='l2', use_idf=True, smooth_idf=True)
    tv_matrix = tv.fit_transform(norm_df)
    print('### tf-idf counts within each text document')
    print(tv_matrix)
    tv_matrix = tv_matrix.toarray()
    print('### tf-idf counts in the array format: one array for one text document')
    print(tv_matrix)
    vocab = tv.get_feature_names_out()
    print('### tf-idf counts in the array format: one array for one text document')
    print(pd.DataFrame(np.round(tv_matrix, 2), columns=vocab))


    # word2vec model
    tokenized_corpus = [wpt.tokenize(doc) for doc in norm_df]
    print(tokenized_corpus)
    # parameters for the word2vec model



    feature_size = 10  # word vector dimensionality
    window_context = 1  # context window size
    min_word_count = 25  # min count to be modeled
    sample = 1e-3  # downsample setting for frequent words
    w2v_model = word2vec.Word2Vec(tokenized_corpus, vector_size =feature_size, window=window_context, min_count=min_word_count,
                                  sample=sample, epochs=100)

    # visualize word embeddings
    words = w2v_model.wv.index_to_key
    wvs = w2v_model.wv[words]
    # More on TSNE: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    tsne = TSNE(n_components=2, random_state=0, n_iter=5000, perplexity=2)
    np.set_printoptions(suppress=True)
    T = tsne.fit_transform(wvs)
    labels = words
    plt.figure(figsize=(12, 6))
    plt.scatter(T[:, 0], T[:, 1], c='orange', edgecolors='r')
    for label, x, y in zip(labels, T[:, 0], T[:, 1]):
        plt.annotate(label, xy=(x + 1, y + 1), xytext=(0, 0), textcoords='offset points')
    plt.show()
    print("### a dense word vector")
    print(w2v_model.wv['fox'])


def avg_word_vectorizer(corpus, model, num_features):
    vocabulary = set(model.wv.index_to_key)
    features = [avg_word_vectors(tokenized_sentence, model, vocabulary, num_features) for tokenized_sentence in corpus]
    return np.array(features)

comments=pd.read_csv(r'comments.csv')
comment=comments.dropna()
commentgroup=comment.groupby(['imdb_id'])['comment_text'].apply(list).to_frame()
commentgroup['comment_text']=commentgroup['comment_text'].apply(lambda x:str(x).replace('[','').replace(']',''))


comment_list = commentgroup['comment_text'].tolist()
comment_text = []
comment_text.append(comment_list[3].split(','))
comment_text=np.array(comment_text)

# text normalization
nltk.download('stopwords')
wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')

normalize_comment_text = np.vectorize(normalize_document)
norm_comment_text = normalize_comment_text(comment_text)
norm_comment_text =norm_comment_text.ravel()
print('### input text data after preprocessing')
print(norm_comment_text)

# bag of words
cv = CountVectorizer(min_df=0., max_df=1.)
cv_matrix1 = cv.fit_transform(norm_comment_text)
print('### word counts within each text document')
print(cv_matrix1)

cv_matrix1 = cv_matrix1.toarray()
print('### word counts in the array format: one array for one text document')
print(cv_matrix1)

vocab = cv.get_feature_names_out()
print('### word counts in a data frame format with columns being the words in the dictionary')
print(pd.DataFrame(cv_matrix1, columns=vocab))
comment_bag_of_word=pd.DataFrame(cv_matrix1, columns=vocab)

# bag of n-grams
# set n-gram range to 1,2 to get both unigrams and bigrams
bv = CountVectorizer(ngram_range=(1, 2))
bv_matrix = bv.fit_transform(norm_comment_text)
print('### unigram and bigram counts within each text document')
print(bv_matrix)
bv_matrix = bv_matrix.toarray()
print('### unigram and bigram counts in the array format: one array for one text document')
print(bv_matrix)
vocab = bv.get_feature_names_out()
print('### unigram and bigram counts in a data frame format with columns being the words in the dictionary')
print(pd.DataFrame(bv_matrix, columns=vocab))
comment_bag_of_ngrams=pd.DataFrame(bv_matrix, columns=vocab)


# tf-idf
print('### tf-idf counts')
print('### Approach 1: CountVectorizer + Tfidftransformer')
tt = TfidfTransformer(norm='l2', use_idf=True)
tt_matrix = tt.fit_transform(cv_matrix1)
print('### tf-idf counts within each text document')
print(tt_matrix)
tt_matrix = tt_matrix.toarray()
print('### tf-idf counts in the array format: one array for one text document')
print(tt_matrix)
vocab = cv.get_feature_names_out()
print('### tf-idf counts in a data frame format with columns being the words in the dictionary')
print(pd.DataFrame(np.round(tt_matrix, 2), columns=vocab))
comment_if_idf=pd.DataFrame(np.round(tt_matrix, 2), columns=vocab)

# word2vec model
tokenized_comment_text = [wpt.tokenize(doc) for doc in norm_comment_text]
print(tokenized_comment_text)
# parameters for the word2vec model
feature_size = 10  # word vector dimensionality
window_context = 1  # context window size
min_word_count = 25  # min count to be modeled
sample = 1e-3  # downsample setting for frequent words
w2v_model = word2vec.Word2Vec(tokenized_comment_text, window=window_context, min_count=min_word_count, sample=sample,epochs=100)


# visualize word embeddings
words = w2v_model.wv.index_to_key
wvs = w2v_model.wv[words]
# More on TSNE: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
tsne = TSNE(n_components=2, random_state=0, n_iter=5000, perplexity=2)
np.set_printoptions(suppress=True)
T = tsne.fit_transform(wvs)
labels = words
plt.figure(figsize=(12, 6))
plt.scatter(T[:, 0], T[:, 1], c='orange', edgecolors='r')
for label, x, y in zip(labels, T[:, 0], T[:, 1]):
    plt.annotate(label, xy=(x + 1, y + 1), xytext=(0, 0), textcoords='offset points')
# plt.show()

import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
from gensim.models import word2vec, Word2Vec
from sklearn.manifold import TSNE
from sklearn.cluster import AffinityPropagation, KMeans
from sklearn.decomposition import PCA

reviews=pd.read_csv(r'reviews.csv')
review=reviews.dropna()
reviewcontent=review.groupby(['imdb_id'])['review_content'].apply(list).to_frame()
reviewcontent['review_content']=reviewcontent['review_content'].apply(lambda x:str(x).replace('[','').replace(']',''))

def normalize_document(doc):
    doc = re.sub(r'[^a-zA-Z\s]', '', doc)
    doc = doc.lower()
    doc = doc.strip()
    tokens = wpt.tokenize(doc)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    doc = ' '.join(filtered_tokens)
    return doc

def avg_word_vectors(words, model, vocabulary, num_features):
    feature_vector = np.zeros((num_features,), dtype="float64")
    nwords = 0.
    for word in words:
        if word in vocabulary:
            nwords = nwords + 1.
            feature_vector = np.add(feature_vector, model[word])  # add the word vector if the word is in the dictionary
    if nwords:
        feature_vector = np.divide(feature_vector, nwords)  # averaging across all words
    return feature_vector

def avg_word_vectorizer(corpus, model, num_features):
    vocabulary = set(model.wv.index_to_key)
    features = [avg_word_vectors(tokenized_sentence, model, vocabulary, num_features) for tokenized_sentence in corpus]
    return np.array(features)

review_content_list = reviewcontent['review_content'].tolist()
review_content = []
review_content.append(review_content_list[0].split(','))
review_content=np.array(review_content)

# text normalization
nltk.download('stopwords')
wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')

normalize_review_content = np.vectorize(normalize_document)
norm_review_content = normalize_review_content(review_content)
norm_review_content =norm_review_content.ravel()
print('### input text data after preprocessing')
print(norm_review_content)

# bag of words
cv = CountVectorizer(min_df=0., max_df=1.)
cv_matrix2 = cv.fit_transform(norm_review_content)
print('### word counts within each text document')
print(cv_matrix2)

cv_matrix2 = cv_matrix2.toarray()
print('### word counts in the array format: one array for one text document')
print(cv_matrix2)

vocab = cv.get_feature_names_out()
print('### word counts in a data frame format with columns being the words in the dictionary')
print(pd.DataFrame(cv_matrix2, columns=vocab))
content_bag_of_words=pd.DataFrame(cv_matrix2, columns=vocab)


# bag of n-grams
# set n-gram range to 1,2 to get both unigrams and bigrams
bv = CountVectorizer(ngram_range=(1, 2))
bv_matrix = bv.fit_transform(norm_review_content)
print('### unigram and bigram counts within each text document')
print(bv_matrix)
bv_matrix = bv_matrix.toarray()
print('### unigram and bigram counts in the array format: one array for one text document')
print(bv_matrix)
vocab = bv.get_feature_names_out()
print('### unigram and bigram counts in a data frame format with columns being the words in the dictionary')
print(pd.DataFrame(bv_matrix, columns=vocab))
content_gan_of_ngrams=pd.DataFrame(bv_matrix, columns=vocab)

# tf-idf
print('### tf-idf counts')
print('### Approach 1: CountVectorizer + Tfidftransformer')
tt = TfidfTransformer(norm='l2', use_idf=True)
tt_matrix = tt.fit_transform(cv_matrix2)
print('### tf-idf counts within each text document')
print(tt_matrix)
tt_matrix = tt_matrix.toarray()
print('### tf-idf counts in the array format: one array for one text document')
print(tt_matrix)
vocab = cv.get_feature_names_out()
print('### tf-idf counts in a data frame format with columns being the words in the dictionary')
print(pd.DataFrame(np.round(tt_matrix, 2), columns=vocab))
content_if_idf=pd.DataFrame(np.round(tt_matrix, 2), columns=vocab)


# word2vec model
tokenized_review_content = [wpt.tokenize(doc) for doc in norm_review_content]
print(tokenized_review_content)
# parameters for the word2vec model
feature_size = 10  # word vector dimensionality
window_context = 1  # context window size
min_word_count = 25  # min count to be modeled
sample = 1e-3  # downsample setting for frequent words
w2v_model = word2vec.Word2Vec(tokenized_review_content, window=window_context, min_count=min_word_count, sample=sample,epochs=100)

# visualize word embeddings
words = w2v_model.wv.index_to_key
wvs = w2v_model.wv[words]
# More on TSNE: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
tsne = TSNE(n_components=2, random_state=0, n_iter=5000, perplexity=2)
np.set_printoptions(suppress=True)
T = tsne.fit_transform(wvs)
labels = words
plt.figure(figsize=(12, 6))
plt.scatter(T[:, 0], T[:, 1], c='orange', edgecolors='r')
for label, x, y in zip(labels, T[:, 0], T[:, 1]):
    plt.annotate(label, xy=(x + 1, y + 1), xytext=(0, 0), textcoords='offset points')
# plt.show()

import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
from gensim.models import word2vec, Word2Vec
from sklearn.manifold import TSNE
from sklearn.cluster import AffinityPropagation, KMeans
from sklearn.decomposition import PCA

reviews=pd.read_csv(r'reviews.csv')
review=reviews.dropna()
reviewheader=review.groupby(['imdb_id'])['review_header'].apply(list).to_frame()
reviewheader['review_header']=reviewheader['review_header'].apply(lambda x:str(x).replace('[','').replace(']',''))

def normalize_document(doc):
    doc = re.sub(r'[^a-zA-Z\s]', '', doc)
    doc = doc.lower()
    doc = doc.strip()
    tokens = wpt.tokenize(doc)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    doc = ' '.join(filtered_tokens)
    return doc

def avg_word_vectors(words, model, vocabulary, num_features):
    feature_vector = np.zeros((num_features,), dtype="float64")
    nwords = 0.
    for word in words:
        if word in vocabulary:
            nwords = nwords + 1.
            feature_vector = np.add(feature_vector, model[word])  # add the word vector if the word is in the dictionary
    if nwords:
        feature_vector = np.divide(feature_vector, nwords)  # averaging across all words
    return feature_vector

def avg_word_vectorizer(corpus, model, num_features):
    vocabulary = set(model.wv.index_to_key)
    features = [avg_word_vectors(tokenized_sentence, model, vocabulary, num_features) for tokenized_sentence in corpus]
    return np.array(features)

review_header_list = reviewheader['review_header'].tolist()
review_header = []
review_header.append(review_header_list[0].split(','))
review_header=np.array(review_header)

# text normalization
nltk.download('stopwords')
wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')

normalize_review_header = np.vectorize(normalize_document)
norm_review_header = normalize_review_header(review_header)
norm_review_header =norm_review_header.ravel()
print('### input text data after preprocessing')
print(norm_review_header)

# bag of words
cv = CountVectorizer(min_df=0., max_df=1.)
cv_matrix = cv.fit_transform(norm_review_header)
print('### word counts within each text document')
print(cv_matrix)

cv_matrix = cv_matrix.toarray()
print('### word counts in the array format: one array for one text document')
print(cv_matrix)

vocab = cv.get_feature_names_out()
print('### word counts in a data frame format with columns being the words in the dictionary')
print(pd.DataFrame(cv_matrix, columns=vocab))
header_bag_of_words=pd.DataFrame(cv_matrix, columns=vocab)

# bag of n-grams
# set n-gram range to 1,2 to get both unigrams and bigrams
bv = CountVectorizer(ngram_range=(1, 2))
bv_matrix = bv.fit_transform(norm_review_header)
print('### unigram and bigram counts within each text document')
print(bv_matrix)
bv_matrix = bv_matrix.toarray()
print('### unigram and bigram counts in the array format: one array for one text document')
print(bv_matrix)
vocab = bv.get_feature_names_out()
print('### unigram and bigram counts in a data frame format with columns being the words in the dictionary')
print(pd.DataFrame(bv_matrix, columns=vocab))
header_bag_of_ngrams=pd.DataFrame(bv_matrix, columns=vocab)

# tf-idf
print('### tf-idf counts')
print('### Approach 1: CountVectorizer + Tfidftransformer')
tt = TfidfTransformer(norm='l2', use_idf=True)
tt_matrix = tt.fit_transform(cv_matrix)
print('### tf-idf counts within each text document')
print(tt_matrix)
tt_matrix = tt_matrix.toarray()
print('### tf-idf counts in the array format: one array for one text document')
print(tt_matrix)
vocab = cv.get_feature_names_out()
print('### tf-idf counts in a data frame format with columns being the words in the dictionary')
print(pd.DataFrame(np.round(tt_matrix, 2), columns=vocab))
header_if_idf=pd.DataFrame(np.round(tt_matrix, 2), columns=vocab)


# word2vec model
tokenized_review_header = [wpt.tokenize(doc) for doc in norm_review_header]
print(tokenized_review_header)
# parameters for the word2vec model
feature_size = 10  # word vector dimensionality
window_context = 1  # context window size
min_word_count = 25  # min count to be modeled
sample = 1e-3  # downsample setting for frequent words
w2v_model = word2vec.Word2Vec(tokenized_review_header, window=window_context, min_count=min_word_count, sample=sample,epochs=100)

# visualize word embeddings
words = w2v_model.wv.index_to_key
wvs = w2v_model.wv[words]
# More on TSNE: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
tsne = TSNE(n_components=2, random_state=0, n_iter=5000, perplexity=2)
np.set_printoptions(suppress=True)
T = tsne.fit_transform(wvs)
labels = words
plt.figure(figsize=(12, 6))
plt.scatter(T[:, 0], T[:, 1], c='orange', edgecolors='r')
for label, x, y in zip(labels, T[:, 0], T[:, 1]):
    plt.annotate(label, xy=(x + 1, y + 1), xytext=(0, 0), textcoords='offset points')
# plt.show()

