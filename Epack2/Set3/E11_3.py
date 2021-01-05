
#%%
root = '/home/tuomas/Python/DATA.STAT.840/Epack2/Set3/'

#%% Load the data
sentences = []
with open(root+'hmm_sentences.txt', 'r') as f:
    line = True
    while(line):
        line = f.readline()
        sentences.append(line)
        
    f.close()

sentences = [s[:-1].split() for s in sentences[:-1]] 

#%%
import gensim
gensim_dictionary = gensim.corpora.Dictionary(sentences)

# Train the word2vec model
word2vecmodel = gensim.models.word2vec.Word2Vec(sentences=sentences, size=5, 
                                                window=3, min_count=1, 
                                                workers=4, sg=0)

#%%
words = ['where', 'dog', 'explain']
for w in words:
    print(w)
    print(word2vecmodel.wv[w])
    print()