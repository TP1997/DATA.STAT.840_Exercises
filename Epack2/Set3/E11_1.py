import os
import sys
sys.path.insert(0,'/home/tuomas/Python/DATA.STAT.840/Epack2/Set3')
import helpers

#%% Load the data
root = '/home/tuomas/Python/DATA.STAT.840/Epack2/Set3/20_newsgroups/'
subd = ['rec.autos/', 'rec.motorcycles/', 'rec.sport.baseball/', 'rec.sport.hockey/']
newsgroup_texts = []
for sd in subd:
    files = os.listdir(root+sd)
    for file in files:
        with open(root + sd + file, encoding='cp1252') as f:
            newsgroup_texts.append(f.read())

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

#%%
a = "%"
sum_=0
for l in range(len(newsgroup_lemmatizedtexts)):
    if a in newsgroup_lemmatizedtexts[l]:
        sum_+=1
        print(l)
        
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

#%% Create a tfidf matrix
vocab_pruned = vocab_unif[pruningdecisions.astype('bool')]
tfidf = helpers.create_tfidf(newsgroup_pruned, vocab_pruned, pruned_vocab_indicies)

#%% Normalize tfidf vectors
for k in range(tfidf.shape[0]):
    tfidf[k,:] /= np.sqrt(np.sum(tfidf[k,:].multiply(tfidf[k,:]),axis=1)[0]+0.0000000001)

#%%
import sklearn, sklearn.decomposition
svdmodel = sklearn.decomposition.TruncatedSVD(n_components=2, n_iter=70, random_state=1)
docplot = svdmodel.fit(tfidf).transform(tfidf)

#%% Plot 2d vectors
import matplotlib.pyplot as plt

figure, axes = plt.subplots();
colors = 'rgbc'
for k in range(len(subd)):
    lo = k*1000
    hi = (k+1)*1000
    axes.scatter(docplot[lo:hi,0], docplot[lo:hi,1], c=colors[k]);

















