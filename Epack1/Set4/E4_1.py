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
book_name = 'The Wonderful Wizard of Oz'
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
book_text = book_texts[0]

#%% Get the paragraphs
# b)
paragraphs = paragraphs=re.split('\n[ \n]*\n', book_text)
paragraphs = paragraphs[1:-1]

#%% Process the paragraphs
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

paragraphs_nltktexts = tokenize(paragraphs)

#%% Make all crawled texts lowercase
paragraphs_lowercasetexts=[]
for k in range(len(paragraphs_nltktexts)):
    temp_lowercasetext=[]
    for l in range(len(paragraphs_nltktexts[k])):
        lowercaseword=paragraphs_nltktexts[k][l].lower()
        temp_lowercasetext.append(lowercaseword)
    temp_lowercasetest=nltk.Text(temp_lowercasetext)
    paragraphs_lowercasetexts.append(temp_lowercasetext)

#%% Stem the loaded texts
stemmer=nltk.stem.porter.PorterStemmer()
def stemtext(nltktexttostem):
    stemmedtext=[]
    for l in range(len(nltktexttostem)):
        # Stem the word
        wordtostem=nltktexttostem[l]
        stemmedword=stemmer.stem(wordtostem)
        # Store the stemmed word
        stemmedtext.append(stemmedword)
    return(stemmedtext)

paragraphs_stemmedtexts=[]
for k in range(len(paragraphs_lowercasetexts)):
    temp_stemmedtext=stemtext(paragraphs_lowercasetexts[k])
    temp_stemmedtext=nltk.Text(temp_stemmedtext)
    paragraphs_stemmedtexts.append(temp_stemmedtext)

#%% POS tag and lemmatize the loaded texts
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

paragraphs_lemmatizedtexts=[]
for k in range(len(paragraphs_stemmedtexts)):
    lemmatizedtext=lemmatizetext(paragraphs_stemmedtexts[k])
    lemmatizedtext=nltk.Text(lemmatizedtext)
    paragraphs_lemmatizedtexts.append(lemmatizedtext)
    
#%% Create unique vocabularities from the ebooks
# Find the vocabulary, in a distributed fashion
import numpy
vocabularies=[]
indices_in_vocabularies=[]
# Find the vocabulary of each document
for k in range(len(paragraphs_lemmatizedtexts)):
    # Get unique words and where they occur
    temptext = paragraphs_lemmatizedtexts[k]
    uniqueresults = numpy.unique(temptext,return_inverse=True)
    uniquewords = uniqueresults[0]
    wordindices = uniqueresults[1]
    # Store the vocabulary and indices of document words in it
    vocabularies.append(uniquewords)
    indices_in_vocabularies.append(wordindices)
    
#%% Create a unified vocabulary from the ebooks
# Unify the vocabularies.
# First concatenate all vocabularies
tempvocabulary = []
for k in range(len(vocabularies)):
    tempvocabulary.extend(vocabularies[k])

# Find the unique elements among all vocabularies
uniqueresults = numpy.unique(tempvocabulary,return_inverse=True)
unifiedvocabulary = uniqueresults[0]
wordindices = uniqueresults[1]

# Translate previous indices to the unified vocabulary.
# Must keep track where each vocabulary started in
# the concatenated one.
vocabularystart = 0
indices_in_unifiedvocabulary = []
for k in range(len(paragraphs_lemmatizedtexts)):
    # In order to shift word indices, we must temporarily
    # change their data type to a Numpy array
    tempindices = numpy.array(indices_in_vocabularies[k])
    tempindices = tempindices + vocabularystart
    tempindices = wordindices[tempindices]
    indices_in_unifiedvocabulary.append(tempindices)
    vocabularystart += len(vocabularies[k])

#%% Count the numbers of occurrences of each unique word
unifiedvocabulary_totaloccurrencecounts=numpy.zeros((len(unifiedvocabulary),1))
unifiedvocabulary_documentcounts=numpy.zeros((len(unifiedvocabulary),1))
# First pass: count occurrences
for k in range(len(paragraphs_lemmatizedtexts)):
    print(k)
    occurrencecounts=numpy.zeros((len(unifiedvocabulary),1))
    for l in range(len(indices_in_unifiedvocabulary[k])):
        occurrencecounts[indices_in_unifiedvocabulary[k][l]] += 1
    unifiedvocabulary_totaloccurrencecounts += occurrencecounts
    unifiedvocabulary_documentcounts += (occurrencecounts>0)

#%% Inspect frequent words
# Sort words by largest total (or mean) occurrence count
highest_totaloccurrences_indices=numpy.argsort(-1*unifiedvocabulary_totaloccurrencecounts,axis=0)

#%% Prune the vocabulary
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
#%% Get indices of documents to remaining words
oldtopruned=[]
tempind=-1
for k in range(len(unifiedvocabulary)):
    if pruningdecisions[k]==0:
        tempind=tempind+1
        oldtopruned.append(tempind)
    else:
        oldtopruned.append(-1)

#%% Create pruned texts
paragraphs_prunedtexts=[]
indices_in_prunedvocabulary=[]
for k in range(len(paragraphs_lemmatizedtexts)):
    print(k)
    temp_newindices=[]
    temp_newdoc=[]
    for l in range(len(paragraphs_lemmatizedtexts[k])):
        temp_oldindex=indices_in_unifiedvocabulary[k][l]
        temp_newindex=oldtopruned[temp_oldindex]
        if temp_newindex!=-1:
            temp_newindices.append(temp_newindex)
            temp_newdoc.append(unifiedvocabulary[temp_oldindex])
    paragraphs_prunedtexts.append(temp_newdoc)
    indices_in_prunedvocabulary.append(temp_newindices)



#%% Create TF-IDF vectors
n_docs=len(paragraphs_prunedtexts)
n_vocab=len(remainingvocabulary)
















