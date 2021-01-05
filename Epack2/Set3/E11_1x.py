import os
#%% Load the data
root = '/home/tuomas/Python/DATA.STAT.840/Epack2/Set3/20_newsgroups/'
subd = ['rec.autos/', 'rec.motorcycles/', 'rec.sport.baseball/']
newsgroup_texts = []
for sd in subd:
    files = os.listdir(root+sd)
    for file in files:
        with open(root + sd + file, encoding='cp1252') as f:
            newsgroup_texts.append(f.read())
            
#%% Tokenize
import nltk
nltk.download('punkt')

def tokenize(texts):
    books_nltktexts = []
    for txt in texts:
        temp = nltk.word_tokenize(txt)
        temp = nltk.Text(temp)
    
        books_nltktexts.append(temp)    
        
    return books_nltktexts

newsgroup_nltktexts = tokenize(newsgroup_texts)

#%% Case removal
def lowercase(texts):
    lowercasetexts = []
    for doc in texts:
        temp_lc = []
        for w in doc:
            temp_lc.append(w.lower())
        temp_lc = nltk.Text(temp_lc)
        lowercasetexts.append(temp_lc)
        
    return lowercasetexts

newsgroup_lctexts = lowercase(newsgroup_nltktexts)

#%% Stemming
stemmer=nltk.stem.porter.PorterStemmer()
def stemtext(text):
    stemmedtext = []
    for wrd in text:
        stemmedword = stemmer.stem(wrd)
        stemmedtext.append(stemmedword)
    
    return(stemmedtext)

def stem(texts):
    stemmedtexts = []
    for txt in texts:
        temp_stemmedtext = stemtext(txt)
        temp_stemmedtext = nltk.Text(temp_stemmedtext)
        stemmedtexts.append(temp_stemmedtext)
        
    return stemmedtexts

newsgroup_stemtexts = stem(newsgroup_lctexts)

#%% Lemmatization
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
lemmatizer = nltk.stem.WordNetLemmatizer()

# Convert a POS tag for WordNet
def tagtowordnet(postag):
    wordnettag=-1
    if postag[0]=='N':
        wordnettag='n'
    elif postag[0]=='V':
        wordnettag='v'
    elif postag[0]=='J':
        wordnettag='a'
    elif postag[0]=='R':
        wordnettag='r'
    return(wordnettag)

def lemmatizetext(text):
    # Tag the text with POS tags
    taggedtext = nltk.pos_tag(text)
    # Lemmatize each word text
    lemmatizedtext = []
    for l in range(len(taggedtext)):
        # Lemmatize a word using the WordNet converted POS tag
        wordtolemmatize = taggedtext[l][0]
        wordnettag = tagtowordnet(taggedtext[l][1])
        if wordnettag != -1:
            lemmatizedword = lemmatizer.lemmatize(wordtolemmatize,wordnettag)
        else:
            lemmatizedword = wordtolemmatize
        lemmatizedtext.append(lemmatizedword)
        
    return lemmatizedtext

def lemmatize(texts):
    lemmatizedtexts = []
    for txt in texts:
        lemmatizedtext = lemmatizetext(txt)
        lemmatizedtext = nltk.Text(lemmatizedtext)
        lemmatizedtexts.append(lemmatizedtext)
        
    return lemmatizedtexts
    
newsgroup_lemtexts = lemmatize(newsgroup_stemtexts)

#%% Create unique vocabularities
import numpy

def get_unique_vocabs(texts):
    vocabularies = []
    indices_in_vocabularies = []
    # Find the vocabulary of each document
    for txt in texts:
        # Get unique words and where they occur
        uniqueresults = numpy.unique(txt,return_inverse=True)
        uniquewords = uniqueresults[0]
        wordindices = uniqueresults[1]
        # Store the vocabulary and indices of document words in it
        vocabularies.append(uniquewords)
        indices_in_vocabularies.append(wordindices)
        
    return vocabularies, indices_in_vocabularies

vocabs_uniq, indicies_uniq = get_unique_vocabs(newsgroup_lemtexts)

#%% Create a unified vocabulary
def get_unified_vocab(vocabs, indicies, lemtexts):
    # Concatenate all vocabularies
    tempvocabulary = []
    for k in range(len(vocabs)):
        tempvocabulary.extend(vocabs[k])    
        
    # Find the unique elements among all vocabularies
    uniqueresults = numpy.unique(tempvocabulary,return_inverse=True)
    unifiedvocabulary = uniqueresults[0]
    wordindices = uniqueresults[1]
    
    # Translate previous indices to the unified vocabulary.
    # Must keep track where each vocabulary started in
    # the concatenated one.
    vocabularystart = 0
    indices_in_unifiedvocabulary = []
    for k in range(len(lemtexts)):
        # In order to shift word indices, we must temporarily
        # change their data type to a Numpy array
        tempindices = numpy.array(indicies[k])
        tempindices = tempindices + vocabularystart
        tempindices = wordindices[tempindices]
        indices_in_unifiedvocabulary.append(tempindices)
        vocabularystart += len(vocabs[k])
        
    return unifiedvocabulary, indices_in_unifiedvocabulary

vocab_unif, indicies_unif = get_unified_vocab(vocabs_uniq, indicies_uniq, newsgroup_lemtexts)

#%% Count the numbers of occurrences of each unique word
def get_occurances(vocab_unif, indicies_unif, lemtexts):
    vocab_unif_totaloccurrencecounts = numpy.zeros((len(vocab_unif),1))
    vocab_unif_documentcounts = numpy.zeros((len(vocab_unif),1))
    # Count occurrences
    for k in range(len(lemtexts)):
        print(k)
        occurrencecounts = numpy.zeros((len(vocab_unif),1))
        for l in range(len(indicies_unif[k])):
            occurrencecounts[indicies_unif[k][l]] += 1
        vocab_unif_totaloccurrencecounts += occurrencecounts
        vocab_unif_documentcounts += (occurrencecounts>0)

    return vocab_unif_totaloccurrencecounts, vocab_unif_documentcounts

totcounts, doccounts = get_occurances(vocab_unif, indicies_unif, newsgroup_lemtexts)

# Sort words by largest total (or mean) occurrence count
highest_totaloccurrences_indices=numpy.argsort(-1*totcounts,axis=0)

#%% Prune the vocabulary
nltk.download('stopwords')
def get_pruningdecisions(vocab_unif):
    nltkstopwords = nltk.corpus.stopwords.words('english')    
    pruningdecisions = numpy.ones((len(vocab_unif)))
    for k in range(len(vocab_unif)):
        # Rule 1: check the nltk stop word list
        if (vocab_unif[k] in nltkstopwords):
            pruningdecisions[k] = 0 # If 0, skip the word
        # Rule 2: if the word is too short
        if len(vocab_unif[k])<2:
            pruningdecisions[k] = 0
        # Rule 3: if the word is too long
        if len(vocab_unif[k])>20:
            pruningdecisions[k] = 0
        # Rule 4: if the word is in the top 1% of frequent words
        if (k in highest_totaloccurrences_indices[0:int(numpy.floor(len(vocab_unif)*0.01))]):
            pruningdecisions[k] = 0
        # Rule 5: if the word is occurring less than 4 times
        if(highest_totaloccurrences_indices[k] < 4):
            pruningdecisions[k] = 0
            
    return pruningdecisions

pruningdecisions = get_pruningdecisions(vocab_unif)

#%% Filter out the unnecessary words in vocabulary
vocab_unif_pruned = vocab_unif[pruningdecisions.astype('bool')]
totcounts_pruned = totcounts[pruningdecisions.astype('bool')]

#%% Get indices of documents to remaining words
oldtopruned=[]
tempind=-1
for k in range(len(vocab_unif)):
    if pruningdecisions[k]==0:
        tempind += 1
        oldtopruned.append(tempind)
    else:
        oldtopruned.append(-1)

#%% Create pruned texts
def get_pruned_texts(vocab_unif, indicies_unif, lemtexts, oldtopruned):
    paragraphs_prunedtexts=[]
    indices_in_prunedvocabulary=[]
    for k in range(len(lemtexts)):
        print(k)
        temp_newindices=[]
        temp_newdoc=[]
        for l in range(len(lemtexts[k])):
            temp_oldindex=indicies_unif[k][l]
            temp_newindex=oldtopruned[temp_oldindex]
            if temp_newindex!=-1:
                temp_newindices.append(temp_newindex)
                temp_newdoc.append(vocab_unif[temp_oldindex])
        paragraphs_prunedtexts.append(temp_newdoc)
        indices_in_prunedvocabulary.append(temp_newindices)
        
    return paragraphs_prunedtexts, indices_in_prunedvocabulary

newsgroup_pruned, pruned_vocab_indicies = get_pruned_texts(vocab_unif, indicies_unif, 
                                                           newsgroup_lemtexts, oldtopruned)

#%% Create TF-IDF vectors
import scipy

n_docs = len(newsgroup_pruned)
n_vocab = len(vocab_unif_pruned)
# Matrix of term frequencies
tfmatrix = scipy.sparse.lil_matrix((n_docs,n_vocab))
# Row vector of document frequencies
dfvector = scipy.sparse.lil_matrix((1,n_vocab))
# Loop over documents
for k in range(n_docs):
    # Row vector of which words occurred in this document
    temp_dfvector = scipy.sparse.lil_matrix((1,n_vocab))
    # Loop over words
    for l in range(len(newsgroup_pruned[k])):
        # Add current word to term-frequency count and document-count
        currentword = pruned_vocab_indicies[k][l]
        tfmatrix[k,currentword] += 1
        temp_dfvector[0,currentword] = 1
    # Add which words occurred in this document to overall document counts
    dfvector = dfvector+temp_dfvector
    
# Use the count statistics to compute the tf-idf matrix
tfidfmatrix = scipy.sparse.lil_matrix((n_docs,n_vocab))
# Let's use raw term count, and smoothed logarithmic idf
idfvector = numpy.log(1+((dfvector+1)**-1)*n_docs)
for k in range(n_docs):
    # Combine the tf and idf terms
    tfidfmatrix[k,:] = tfmatrix[k,:]*idfvector
    
#%%
    



    




























