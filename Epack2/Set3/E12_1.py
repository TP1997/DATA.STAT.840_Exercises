import os
import sys
sys.path.insert(0,'/home/tuomas/Python/DATA.STAT.840/Epack2/Set3')
import helpers

#%% Load the data
root = '/home/tuomas/Python/DATA.STAT.840/Epack2/Set3/20_newsgroups/'
subd = ['rec.autos/', 'rec.motorcycles/', 'rec.sport.baseball/', 'rec.sport.hockey/']
newsgroup_texts = []
filenames = []
for sd in subd:
    os.chdir(root+sd)
    files = os.listdir()
    for file in files:
        with open(file, encoding='cp1252') as f:
            newsgroup_texts.append(f.read())
    filenames.append(files)

#%% Remove metainformation?

#%% Tokenize
import nltk
nltk.download('punkt')
newsgroup_nltktexts = helpers.tokenize(newsgroup_texts)

#%% Case removal
newsgroup_lowercasetexts = helpers.lowercase(newsgroup_nltktexts)

#%% Stemming
newsgroup_stemtexts = helpers.stem(newsgroup_lowercasetexts)

#%% Lemmatization
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
newsgroup_lemmatizedtexts = helpers.lemmatize(newsgroup_stemtexts)

#%% Create unique vocabularities
vocabs_uniq, indicies_uniq = helpers.get_unique_vocabs(newsgroup_lemmatizedtexts)

#%% Create a unified vocabulary
vocab_unif, indicies_unif = helpers.get_unified_vocab(vocabs_uniq, indicies_uniq, newsgroup_lemmatizedtexts)            

#%% Count the numbers of occurrences of each unique word
import numpy as np
totcounts, doccounts = helpers.get_occurances(vocab_unif, 
                                              indicies_unif, 
                                              newsgroup_lemmatizedtexts)

highest_totaloccurrences_indices = np.argsort(-1*totcounts,axis=0)

#%% Prune the vocabulary
nltk.download('stopwords')
pruningdecisions = helpers.get_pruningdecisions(vocab_unif, highest_totaloccurrences_indices)

#%% Get indices of documents to remaining words
oldtopruned=[]
tempind=-1
for k in range(len(vocab_unif)):
    if pruningdecisions[k]==1:
        tempind += 1
        oldtopruned.append(tempind)
    else:
        oldtopruned.append(-1)

#%% Create pruned texts
newsgroup_pruned, pruned_vocab_indicies = helpers.get_pruned_texts(vocab_unif, 
                                                                   indicies_unif, 
                                                                   newsgroup_lemmatizedtexts, 
                                                                   oldtopruned) 

#%% Create a tagged version of each document
import gensim
gensim_tagged_docs = []
for k in range(len(newsgroup_pruned)):
    doctag = 'doc'+str(k)
    tagged_document = gensim.models.doc2vec.TaggedDocument(newsgroup_pruned[k], [doctag])
    gensim_tagged_docs.append(tagged_document)
    
#%% Create a dictionary from the documents !!!NOT NEEDED!!!
gensim_dictionary = gensim.corpora.Dictionary(np.array(gensim_tagged_docs)[:,0])

#%% Train the word2vec model
doc2vecmodel = gensim.models.doc2vec.Doc2Vec(gensim_tagged_docs,vector_size=10, 
                                             window=5, min_count=1, 
                                             workers=4, dm_concat=0)


#%% Find the closest documents
from scipy.spatial.distance import cdist
docs = ['101551','103118','98657','52550']
docidx_local = []
docidx_global = []
for i in range(len(docs)):
    idx = np.where(np.array(filenames[i]) == docs[i])[0][0]
    docidx_local.append(idx)
    docidx_global.append(idx + i*len(filenames[i]))

closest_idx = []
all_docvecs = doc2vecmodel.docvecs.vectors_docs
for idx in docidx_global:
    vec = all_docvecs[idx].reshape(1, all_docvecs[idx].shape[0])
    distances = cdist(vec, all_docvecs).ravel()
    distances[idx] = np.finfo('float').max
    closest_idx.append(np.argmin(distances))
    
#%% Print the results
for i in range(len(docs)):
    closestdoc = np.array(filenames).ravel()[closest_idx[i]]
    print('Closest to {}: {}'.format(docs[i], closestdoc))
    
































