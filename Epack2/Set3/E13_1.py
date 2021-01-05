import requests
import bs4
import sys
sys.path.insert(0,'/home/tuomas/Python/DATA.STAT.840/Epack2/Set3')
import helpers

root = '/home/tuomas/Python/DATA.STAT.840/Epack2/Set3/'
base_url = 'https://www.gutenberg.org'

#%%
gutenberg_response_html = requests.get(base_url + '/browse/scores/top')
gutenberg_parsed = bs4.BeautifulSoup(gutenberg_response_html.content, 'html.parser')
location = gutenberg_parsed.find(id='books-last30')
book_listing_tag = location.next_sibling.next_sibling

#%% Download data from web & save to local filesystem
book_name = 'Pride and Prejudice'
bookdata = helpers.get_bookdata_byname(book_listing_tag, book_name, base_url)
helpers.complete_urls(bookdata)
helpers.download_books(bookdata, root)
# Download data from local filesystem
book_texts = helpers.load_data_local(bookdata, root)

#%% Remove metainformation
book_texts = helpers.remove_headers(book_texts)

#%% Tokenize
import nltk
nltk.download('punkt')
books_nltktexts = helpers.tokenize(book_texts)

#%% Case removal
books_lowercasetexts = helpers.lowercase(books_nltktexts)

#%% Stemming
books_stemtexts = helpers.stem(books_lowercasetexts)

#%% Lemmatization
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
books_lemmatizedtexts = helpers.lemmatize(books_stemtexts)

#%% Create (unique & unified) vocabularities
vocabs_uniq, indicies_uniq = helpers.get_unique_vocabs(books_lemmatizedtexts)
vocab_unif = vocabs_uniq[0]
indicies_unif = indicies_uniq #[0]

#%% Count the numbers of occurrences of each unique word
import numpy as np
totcounts, doccounts = helpers.get_occurances(vocab_unif, 
                                              indicies_unif, 
                                              books_lemmatizedtexts)

highest_totaloccurrences_indices = np.argsort(-1*totcounts,axis=0)
#%% Prune the vocabulary
nltk.download('stopwords')
pruningdecisions = helpers.get_pruningdecisions(vocab_unif, highest_totaloccurrences_indices)

#%% Filter out the unnecessary words in vocabulary
vocab_unif_pruned = vocab_unif[pruningdecisions.astype('bool')]
totcounts_pruned = totcounts[pruningdecisions.astype('bool')]


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
books_pruned, pruned_vocab_indicies = helpers.get_pruned_texts(vocab_unif, 
                                                               indicies_unif, 
                                                               books_lemmatizedtexts, 
                                                               oldtopruned)
books_pruned = books_pruned[0]
pruned_vocab_indicies = pruned_vocab_indicies[0]

#%% Create the LSTM model
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
import keras.callbacks
from keras.utils import to_categorical
import keras.utils.np_utils

model = Sequential()

lstm_layer_size = 50
text_windowsize = 10
features_per_input_sample = 1
model.add(LSTM(lstm_layer_size,
               activation = 'relu',
               input_shape = (text_windowsize, features_per_input_sample)))

output_layer_size = len(vocab_unif)
model.add(Dense(output_layer_size,
                activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

#%% Create a training data set:
n_trainingsamples = len(books_pruned) - text_windowsize #len(books_pruned[0]) - text_windowsize #???
x_data = np.zeros((n_trainingsamples, text_windowsize,1))
y_data = np.zeros((n_trainingsamples))
temp_wordindices = pruned_vocab_indicies #indicies_unif[0] #???
for windowposition in range(n_trainingsamples):
    for k in range(text_windowsize):
        temp_wordindex = temp_wordindices[windowposition+k]
        x_data[windowposition, k, 0] = temp_wordindex
    temp_wordindex = temp_wordindices[windowposition + text_windowsize]
    y_data[windowposition] = temp_wordindex

y_onehotencoded = to_categorical(y_data, num_classes=len(vocab_unif))

#%% Fit the model
model.fit(x=x_data, y=y_onehotencoded, batch_size=100, epochs=10)

#%% Predict new text
n_newtext = 100
start_textwindow = 2000 #???
generated_wordindices = np.squeeze(np.array(x_data[start_textwindow,:,:])) #???
for k in range(n_newtext):
    temp_textwindow = generated_wordindices[k:(k+text_windowsize)]
    temp_probs = model.predict(np.reshape(temp_textwindow, (1,text_windowsize,1)))
    best_word = np.argmax(temp_probs)
    generated_wordindices = np.append(generated_wordindices, best_word)

generated_wordindices = generated_wordindices.astype(int)
newtext = vocab_unif[generated_wordindices]

print(newtext)





