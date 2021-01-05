import requests
import bs4
import re
import fnmatch

# a)
root = '/home/tuomas/Python/DATA.STAT.840/Set2/'
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

#%% Download top 20 books. Part 1
bookdata = get_bookdata(book_listing_tag, 20)
complete_urls(bookdata)
download_books(bookdata)

#%% Download top 20 books. Part 2
# Download data from local filesystem

def load_data_local(bookdata, fileloc):
    book_text = []
    for data in bookdata:
        print("Loading book {}".format(data['id']))
        try:
            with open(fileloc + data["fname"], 'r') as f:
                book_text.append(f.read())
        except:
            with open(fileloc + data["fname"], 'r', encoding='ISO-8859-1') as f:
                book_text.append(f.read())
        
    
    return book_text
    

book_texts = load_data_local(bookdata, root + 'txtfiles/')

#%% Remove metainformation

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

#%% Tokenize loaded texts and change them to NLTK format
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

#%% Lemmatize the loaded texts
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
for k in range(len(books_nltktexts)):
    print('Book {}'.format(k+1))
    lemmatizedtext=lemmatizetext(books_nltktexts[k])
    lemmatizedtext=nltk.Text(lemmatizedtext)
    books_lemmatizedtexts.append(lemmatizedtext)

#%% Create unique vocabularities from the ebooks
# Find the vocabulary, in a distributed fashion
import numpy
vocabularies=[]
indices_in_vocabularies=[]
# Find the vocabulary of each document
for k in range(len(books_lemmatizedtexts)):
    print('Book {}'.format(k+1))
    # Get unique words and where they occur
    temptext = books_lemmatizedtexts[k]
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
indices_in_unifiedvocabulary = []
for k in range(len(books_lemmatizedtexts)):
    print('Book {}'.format(k+1))
    # In order to shift word indices, we must temporarily
    # change their data type to a Numpy array
    tempindices = numpy.array(indices_in_vocabularies[k])
    tempindices = tempindices + vocabularystart
    tempindices = wordindices[tempindices]
    indices_in_unifiedvocabulary.append(tempindices)
    vocabularystart += len(vocabularies[k])

#%% Count the numbers of occurrences of each unique word
unifiedvocabulary_totaloccurrencecounts=numpy.zeros((len(unifiedvocabulary),1))
# Count occurances
for k in range(len(books_lemmatizedtexts)):
    occurrencecounts=numpy.zeros((len(unifiedvocabulary),1))
    for l in range(len(indices_in_unifiedvocabulary[k])):
        occurrencecounts[indices_in_unifiedvocabulary[k][l]] += 1
    unifiedvocabulary_totaloccurrencecounts += occurrencecounts

unifiedvocabulary_totaloccurrencecounts = unifiedvocabulary_totaloccurrencecounts.ravel()

#%% Calculate total count of all words
tot_count = 0
for book in books_lemmatizedtexts:
    tot_count += len(book)

#%% Plot the results
import numpy as np
import matplotlib.pyplot as plt

uvoc_counts = unifiedvocabulary_totaloccurrencecounts
freq = uvoc_counts / tot_count
freq = np.sort(freq)[::-1][:100]
# Zipf's law frequency distribution
a=1.2
ranks = (np.array(list(range(len(uvoc_counts))))+1).astype('int')[:100]
zipf = 1/ranks**(a) / np.sum(1/ranks**(a))
# Plot
real_freq_plt = plt.plot(ranks, freq)
zipf_plt = plt.plot(zipf)

plt.legend([real_freq_plt[0], zipf_plt[0]], ['Frequency', 'Zipf where a={}'.format(a)])
