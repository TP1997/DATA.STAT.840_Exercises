import requests
import bs4
import re
import fnmatch
import nltk
import numpy
import scipy

def get_bookdata(book_listing, k, base_url):
    data = []
    for a_tag in book_listing.find_all('a')[:k]:
        book_name = re.match(r'(.*)(\(\d+\))', a_tag.text)
        book_name = book_name.group(1).strip()
    
        book_id = re.match(r'/ebooks/(\d+)', a_tag.get('href'))
        book_id = book_id.group(1)
    
        book_url = base_url + '/' + book_id + '/'
        
        data.append({"id":book_id, "name":book_name, "url":book_url, "fname":''})
    
    return data

def get_bookdata_byname(book_listing, name, base_url):
    data = []
    for a_tag in book_listing.find_all('a'):
        book_name = re.match(r'(.*)(\(\d+\))', a_tag.text)
        book_name = book_name.group(1).strip()
        
        if name not in book_name: continue
        
        book_id = re.match(r'/ebooks/(\d+)', a_tag.get('href'))
        book_id = book_id.group(1)
        
        book_url = base_url + '/' + book_id + '/'
        
        data.append({"id":book_id, "name":book_name, "url":book_url, "fname":''})
        
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

def download_books(bookdata, root):
    for data in bookdata:
        print("Downloading book: {}".format(data["name"]))
        print("From source: {}".format(data["url"]))
        response = requests.get(data["url"], allow_redirects=True)
        with open(root + data["fname"], 'wb') as f:
            f.write(response.content) 

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

# Tokenize
def tokenize(texts):
    books_nltktexts = []
    for txt in texts:
        temp = nltk.word_tokenize(txt)
        temp = nltk.Text(temp)
    
        books_nltktexts.append(temp)    
        
    return books_nltktexts

# Case removal
def lowercase(texts):
    lowercasetexts = []
    for doc in texts:
        temp_lc = []
        for w in doc:
            temp_lc.append(w.lower())
        temp_lc = nltk.Text(temp_lc)
        lowercasetexts.append(temp_lc)
        
    return lowercasetexts

# Stemming
def stemtext(text):
    stemmer=nltk.stem.porter.PorterStemmer()
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

# Lemmatization
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
    lemmatizer = nltk.stem.WordNetLemmatizer()
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

# Create unique vocabularities
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

# Create a unified vocabulary
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
        
    return unifiedvocabulary, numpy.array(indices_in_unifiedvocabulary)

# Count the numbers of occurrences of each unique word
def get_occurances(vocab_unif, indicies_unif, lemtexts):
    vocab_unif_totaloccurrencecounts = numpy.zeros((len(vocab_unif),1))
    vocab_unif_documentcounts = numpy.zeros((len(vocab_unif),1))
    # Count occurrences
    for k in range(len(lemtexts)):
        print(k)
        occurrencecounts = numpy.zeros((len(vocab_unif),1))
        print('SHAPE: ',occurrencecounts.shape)
        for l in range(len(indicies_unif[k])):
            occurrencecounts[indicies_unif[k][l]] += 1
        vocab_unif_totaloccurrencecounts += occurrencecounts
        vocab_unif_documentcounts += (occurrencecounts>0)

    return vocab_unif_totaloccurrencecounts, vocab_unif_documentcounts

# Prune the vocabulary
def get_pruningdecisions(vocab_unif, occurance_indices):
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
        if (k in occurance_indices[0:int(numpy.floor(len(vocab_unif)*0.01))]):
            pruningdecisions[k] = 0
        # Rule 5: if the word is occurring less than 4 times
        if(occurance_indices[k] < 4):
            pruningdecisions[k] = 0
            
    return pruningdecisions

# Create pruned texts
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

# Create TF-IDF vectors
# Tf = Boolean
# idf = Smoothed logarithmic inverse document frequency
def create_tfidf(pruned_texts, pruned_vocab, pruned_vocab_indicies):
    n_docs = len(pruned_texts)
    n_vocab = len(pruned_vocab)
    # Matrix of term frequencies
    tfmatrix = scipy.sparse.lil_matrix((n_docs,n_vocab))
    # Row vector of document frequencies
    dfvector = scipy.sparse.lil_matrix((1,n_vocab))
    # Loop over documents
    for k in range(n_docs):
        # Row vector of which words occurred in this document
        temp_dfvector = scipy.sparse.lil_matrix((1,n_vocab))
        # Loop over words
        for l in range(len(pruned_texts[k])):
            # Add current word to term-frequency count and document-count
            currentword = pruned_vocab_indicies[k][l]
            tfmatrix[k,currentword] += 1
            temp_dfvector[0,currentword] = 1
        # Add which words occurred in this document to overall document counts
        dfvector += temp_dfvector
    
    # Use the count statistics to compute the tf-idf matrix
    tfidfmatrix = scipy.sparse.lil_matrix((n_docs,n_vocab))
    # Let's use raw term count, and smoothed logarithmic idf
    idfvector = numpy.log(1+((dfvector.toarray()+1)**-1)*n_docs)
    for k in range(n_docs):
        # Combine the tf and idf terms
        #print('TF', tfmatrix[k,:].shape, type(tfmatrix[k,:]))
        #print('IDF', idfvector.shape, type(idfvector))
        #print('TFIDF', tfidfmatrix[k,:].shape, type(tfidfmatrix[k,:]))
        tfidfmatrix[k,:] = tfmatrix[k,:].toarray()*idfvector
        
    print('TFIDF', type(tfidfmatrix[k,:]))
    return tfidfmatrix

