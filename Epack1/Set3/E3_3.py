import requests
import bs4
import re
import fnmatch

root = '/home/tuomas/Python/DATA.STAT.840/Set3/'
base_url = 'https://www.gutenberg.org'

#%%
gutenberg_response_html = requests.get(base_url + '/browse/scores/top')
gutenberg_parsed = bs4.BeautifulSoup(gutenberg_response_html.content, 'html.parser')
location = gutenberg_parsed.find(id='books-last30')
book_listing_tag = location.next_sibling.next_sibling

#%%
def get_bookdata(book_listing, k):
    data = []
    for a_tag in book_listing.find_all('a')[:k]:
        book_name = re.match(r'(.*)(\(\d+\))', a_tag.text)
        book_name = book_name.group(1).strip()
    
        book_id = re.match(r'/ebooks/(\d+)', a_tag.get('href'))
        book_id = book_id.group(1)
    
        book_url = base_url + '/' + book_id + '/'
        
        data.append({"id":book_id, "name":book_name, "url":book_url, "fname":''})
    
    return data

def get_bookdata_byname(book_listing, name):
    data = []
    for a_tag in book_listing.find_all('a'):
        book_name = re.match(r'(.*)(\(\d+\))', a_tag.text)
        book_name = book_name.group(1).strip()
        
        if name not in book_name: continue
        
        book_id = re.match(r'/ebooks/(\d+)', a_tag.get('href'))
        book_id = book_id.group(1)
        
        book_url = base_url + '/' + book_id + '/'
        
        data.append({"id":book_id, "name":book_name, "url":book_url, "fname":''})
        break
    
    return data
        
def complete_urls(bookdata):
    for i in range(len(bookdata)):
        print("Searching txt-file for book with id {}".format(bookdata[i]["id"]))
        indexurl = bookdata[i]["url"]
        
        bookindex_html = requests.get(indexurl)
        bookindex_parsed =  bs4.BeautifulSoup(bookindex_html.content,'html.parser')
        bookindex_links = bookindex_parsed.find_all('a')
        bookindex_hrefs = [bil['href'] for bil in bookindex_links]
        
        book_filenames = [ bih for bih in bookindex_hrefs if fnmatch.fnmatch(bih, '*.txt') ] #.*.txt
        
        bookdata[i]["url"] += book_filenames[0]
        bookdata[i]["fname"] = book_filenames[0]
        
    return

def download_books(bookdata):
    for data in bookdata:
        print("Downloading book: {}".format(data["name"]))
        print("From source: {}".format(data["url"]))
        response = requests.get(data["url"], allow_redirects=True)
        with open(root + 'txtfiles/' + data["fname"], 'wb') as f:
            f.write(response.content)
            
#%%
book_name = 'The Call of the Wild'
bookdata = get_bookdata_byname(book_listing_tag, book_name)
complete_urls(bookdata)
download_books(bookdata)

#%%
# Download data from local filesystem

def load_data_local(bookdata, fileloc):
    book_text = []
    for data in bookdata:
        try:
            with open(fileloc + data["fname"], 'r') as f:
                book_text.append(f.read())
        except:
            with open(fileloc + data["fname"], 'r', encoding='ISO-8859-1') as f:
                book_text.append(f.read()) 
        print("Book {} loaded.".format(data['id']))
    
    return book_text
    
book_texts = load_data_local(bookdata, root + 'txtfiles/')

#%%
# Remove metainformation

def remove_headers(book_texts):
    start_header = '*** START'
    end_header = '*** END'
    new_texts = []
    for text in book_texts:
        start_loc = text.find(start_header)
        print(start_loc)
        start_loc = text[start_loc:].find('\n') + start_loc
        print(start_loc)
        end_loc = text.find(end_header)
        text = text[start_loc : end_loc]
        new_texts.append(text)
        
    return new_texts
        
book_texts = remove_headers(book_texts)

#%%
# Tokenize loaded texts and change them to NLTK format

import nltk
nltk.download('punkt')

def tokenize(texts):
    books_nltktexts = []
    for txt in texts:
        temp = nltk.word_tokenize(txt)
        temp = nltk.Text(temp)
    
        books_nltktexts.append(temp)    
        
    return books_nltktexts

books_nltktexts = tokenize(book_texts)

#%%
# Lemmatize the loaded texts

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
lemmatizer=nltk.stem.WordNetLemmatizer()

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

def lemmatizetext(nltktexttolemmatize):
    # Tag the text with POS tags
    taggedtext = nltk.pos_tag(nltktexttolemmatize)
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
        # Store the lemmatized word
        lemmatizedtext.append(lemmatizedword)
        
    return(lemmatizedtext)

books_lemmatizedtexts=[]
for book_nltktext in books_nltktexts:
    lemmatizedtext=lemmatizetext(book_nltktext)
    lemmatizedtext=nltk.Text(lemmatizedtext)
    books_lemmatizedtexts.append(lemmatizedtext)
    
#%%
# Create unique vocabularities from the ebooks

import numpy
# Find the vocabulary, in a distributed fashion
vocabularies=[]
indices_in_vocabularies=[]
# Find the vocabulary of each document
for book_lemmatizedtext in books_lemmatizedtexts:
    # Get unique words and where they occur
    temptext = book_lemmatizedtext
    uniqueresults = numpy.unique(temptext,return_inverse=True)
    uniquewords = uniqueresults[0]
    wordindices = uniqueresults[1]
    
    # Store the vocabulary and indices of document words in it
    vocabularies.append(uniquewords)
    indices_in_vocabularies.append(wordindices)

#%%
# Create a unified vocabulary from the ebooks
# Unify the vocabularies.
# First concatenate all vocabularies
tempvocabulary = []
for k in range(len(books_lemmatizedtexts)):
    tempvocabulary.extend(vocabularies[k])
    
# Find the unique elements among all vocabularies
uniqueresults = numpy.unique(tempvocabulary,return_inverse=True)
unifiedvocabulary = uniqueresults[0]
wordindices = uniqueresults[1]

# Translate previous indices to the unified vocabulary.
# Must keep track where each vocabulary started in
# the concatenated one.
vocabularystart = 0
myindices_in_unifiedvocabulary = []
for k in range(len(books_lemmatizedtexts)):
    # In order to shift word indices, we must temporarily
    # change their data type to a Numpy array
    tempindices = numpy.array(indices_in_vocabularies[k])
    tempindices = tempindices + vocabularystart
    tempindices = wordindices[tempindices]
    myindices_in_unifiedvocabulary.append(tempindices)
    vocabularystart += len(vocabularies[k])

#%% Count the numbers of occurrences of each unique word...
unifiedvocabulary_totaloccurrencecounts=numpy.zeros((len(unifiedvocabulary),1))
# Count occurances
for k in range(len(books_lemmatizedtexts)):
    occurrencecounts=numpy.zeros((len(unifiedvocabulary),1))
    for l in range(len(myindices_in_unifiedvocabulary[k])):
        occurrencecounts[myindices_in_unifiedvocabulary[k][l]] += 1
        unifiedvocabulary_totaloccurrencecounts += occurrencecounts

#%% Inspect frequent words
# Sort words by largest total (or mean) occurrence count
highest_totaloccurrences_indices = numpy.argsort(-1*unifiedvocabulary_totaloccurrencecounts,axis=0)
print('top-100 words')
print(numpy.squeeze(unifiedvocabulary[highest_totaloccurrences_indices[1:100]]))
print(numpy.squeeze(unifiedvocabulary_totaloccurrencecounts[highest_totaloccurrences_indices[1:100]]))

#%% Vocabulary pruning
# download the stopword list if you do not have it already
nltk.download('stopwords')
nltkstopwords = nltk.corpus.stopwords.words('english')
pruningdecisions = numpy.ones((len(unifiedvocabulary)))
for k in range(len(unifiedvocabulary)):
    # Rule 1: check the nltk stop word list
    if (unifiedvocabulary[k] in nltkstopwords):
        pruningdecisions[k]=0 # If 0, skip the word
    # Rule 2: if the word is too short
    if len(unifiedvocabulary[k])<2:
        pruningdecisions[k]=0
    # Rule 3: if the word is too long
    if len(unifiedvocabulary[k])>20:
        pruningdecisions[k]=0
    # Rule 4: if the word is in the top 1% of frequent words
    if (k in highest_totaloccurrences_indices[0:int(numpy.floor(len(unifiedvocabulary)*0.01))]):
        pruningdecisions[k]=0
    # Rule 5: if the word is occurring less than 4 times
    if(highest_totaloccurrences_indices[k] < 4):
        pruningdecisions[k]=0
        
#%% Filter out the unnecessary words in vocabulary
prunedvocabulary = unifiedvocabulary[pruningdecisions.astype('bool')]
prunedvocabulary_totaloccurrencecounts = unifiedvocabulary_totaloccurrencecounts[pruningdecisions.astype('bool')]

#%% Inspect top words again
highest_totaloccurrences_indices = numpy.argsort(-1*prunedvocabulary_totaloccurrencecounts,axis=0)
print('top-100 words')
print(numpy.squeeze(prunedvocabulary[highest_totaloccurrences_indices[1:100]]))
print(numpy.squeeze(prunedvocabulary_totaloccurrencecounts[highest_totaloccurrences_indices[1:100]]))

#%%
""" PART b) START """
#%% Get indices of documents to remaining words
oldtopruned=[]
tempind=-1
for k in range(len(unifiedvocabulary)):
    if pruningdecisions[k]==1:
        tempind=tempind+1
        oldtopruned.append(tempind)
    else:
        oldtopruned.append(-1)

#%% Create pruned texts
books_prunedtexts=[]
myindices_in_prunedvocabulary=[]
for k in range(len(books_lemmatizedtexts)):
    print(k)
    temp_newindices=[]
    temp_newdoc=[]
    for l in range(len(books_lemmatizedtexts[k])):
        temp_oldindex=myindices_in_unifiedvocabulary[k][l]
        temp_newindex=oldtopruned[temp_oldindex]
        if temp_newindex!=-1:
            temp_newindices.append(temp_newindex)
            temp_newdoc.append(unifiedvocabulary[temp_oldindex])
        books_prunedtexts.append(temp_newdoc)
        myindices_in_prunedvocabulary.append(temp_newindices)

#%% Compute statistics of word distances
# Compute counts and subs of distances and squared distances
import scipy
n_vocab=len(prunedvocabulary)
distanceoccurrences=scipy.sparse.lil_matrix((n_vocab,n_vocab))
sumdistances=scipy.sparse.lil_matrix((n_vocab,n_vocab))
sumabsdistances=scipy.sparse.lil_matrix((n_vocab,n_vocab))
sumdistancesquares=scipy.sparse.lil_matrix((n_vocab,n_vocab))

for l in range(len(books_lemmatizedtexts)):
    print(l)
    latestoccurrencepositions=scipy.sparse.lil_matrix((n_vocab,n_vocab))
    # Loop through all word positions m of document l
    for m in range(len(books_prunedtexts[l])):
        # Get the vocabulary index of the current word in position m
        currentword=myindices_in_prunedvocabulary[l][m]
        # Loop through previous words, counting back up to 10 words from current word
        windowsize=min(m,10)
        for n in range(windowsize):
            # Get the vocabulary index of the previous word in position m-n-1
            previousword=myindices_in_prunedvocabulary[l][m-n-1]
            # Is this the fist time we have encountered this word while
            # counting back from the word at m? Then it is the closest pair
            if latestoccurrencepositions[currentword,previousword]<m:
                # Store the occurrence of this word pair with the word at m as the 1st word
                distanceoccurrences[currentword,previousword] += 1
                sumdistances[currentword,previousword]        += ((m-n-1)-m)
                sumabsdistances[currentword,previousword]     += abs((m-n-1)-m)
                sumdistancesquares[currentword,previousword]  += ((m-n-1)-m)**2
                # Store the occurrence of this word pair with the word at n as the 1st word
                distanceoccurrences[previousword,currentword] += 1
                sumdistances[previousword,currentword]        += (m-(m-n-1))
                sumabsdistances[previousword,currentword]     += abs(m-(m-n-1))
                sumdistancesquares[previousword,currentword]  += (m-(m-n-1))**2
                # Mark that we found this pair while counting down from m,
                # so we do not count more distant occurrences of the pair
                latestoccurrencepositions[currentword,previousword]=m
                latestoccurrencepositions[previousword,currentword]=m

#%% Compute distribution statistics based on the counts
distancemeans=scipy.sparse.lil_matrix((n_vocab,n_vocab))
absdistancemeans=scipy.sparse.lil_matrix((n_vocab,n_vocab))
distancevariances=scipy.sparse.lil_matrix((n_vocab,n_vocab))
absdistancevariances=scipy.sparse.lil_matrix((n_vocab,n_vocab))
for m in range(n_vocab):
    print(m)
    # Find the column indices that have at least two occurrences
    tempindices=numpy.nonzero(distanceoccurrences[m,:]>1)[1]
    # The occurrence vector needs to be a non-sparse data type
    tempoccurrences=distanceoccurrences[m,tempindices].todense()
    # Estimate mean of m-n distance
    distancemeans[m,tempindices]=numpy.squeeze(numpy.array(sumdistances[m,tempindices]/tempoccurrences))
    absdistancemeans[m,tempindices]=numpy.squeeze(numpy.array(sumabsdistances[m,tempindices]/tempoccurrences))
    # Estimate variance of m-n distance
    meanterm=distancemeans[m,tempindices].todense()
    meanterm=numpy.multiply(meanterm,meanterm)
    meanterm=numpy.multiply(tempoccurrences/(tempoccurrences-1),meanterm)
    distancevariances[m,tempindices]=numpy.squeeze(\
                            numpy.array(sumdistancesquares[m,tempindices]/(tempoccurrences-1) - meanterm))
    meanterm=absdistancemeans[m,tempindices].todense()
    meanterm=numpy.multiply(meanterm,meanterm)
    meanterm=numpy.multiply(tempoccurrences/(tempoccurrences-1),meanterm)
    absdistancevariances[m,tempindices]=numpy.squeeze(\
                            numpy.array(sumdistancesquares[m,tempindices]/(tempoccurrences-1) - meanterm))
        
#%% Compute overall distance distribution
overalldistancecount=numpy.sum(distanceoccurrences)

overalldistancesum=numpy.sum(sumdistances)

overallabsdistancesum=numpy.sum(sumabsdistances)

overalldistancesquaresum=numpy.sum(sumdistancesquares)

overalldistancemean=overalldistancesum/overalldistancecount

overallabsdistancemean=overallabsdistancesum/overalldistancecount

overalldistancevariance=overalldistancesquaresum/(overalldistancecount-1)\
                        -overalldistancecount/(overalldistancecount-1)*overalldistancemean
                        
overallabsdistancevariance=overalldistancesquaresum/(overalldistancecount-1)\
                        -overalldistancecount/(overalldistancecount-1)*overallabsdistancemean

#%%
""" PART c) START """
#%%






















