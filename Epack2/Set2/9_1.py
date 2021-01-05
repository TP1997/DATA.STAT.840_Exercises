import numpy as np

#%%
vocab = ['a','am','and','answer','are','bird','but','cat','cautious',
         'explain','fox','friendly','help','i','insightful','is','may',
         'robot','strong','the','they','when','will','wise','you']
vocab = np.array(vocab)

doc = ['the robot is insightful but you are strong and i may answer and the wise fox is insightful and you are insightful and i am insightful but i will explain the insightful bird',
       'the bird is insightful',
       'when will they explain the friendly insightful strong insightful bird and is the bird strong and is a strong robot insightful',
       'a cat is strong but you are cautious and i may help but a fox is insightful but are they strong and when may you answer',
       'insightful bird',
       ]
doc = np.array([np.array(dd.split()) for dd in doc])
query = np.array('insightful bird'.split())
#%% TF vectors for documents
tfvec = []
for i in range(doc.shape[0]):
    tmplist = []
    for j in range(vocab.shape[0]):
        tmp = vocab[j] == doc[i]
        tmplist.append(np.sum(tmp))
    tfvec.append(np.array(tmplist))
    
tfvec = np.array(tfvec)
   
#%% IDF vector for documents
idf = np.zeros(25)
for i in range(vocab.shape[0]):
    for d in doc:
        idf[i] += vocab[i] in d
  
T = doc.shape[0]
idf = np.log(T/idf)

#%% TF-IDF vectors for documents
tfidf = tfvec*idf

#%% Cosine similarities
from numpy.linalg import norm

def cos_sim(x1, x2):
    return (np.inner(x1, x2) / (norm(x1)*norm(x2)))

qvec = tfidf[-1]
q_d1_sim = cos_sim(tfidf[0], qvec)
q_d2_sim = cos_sim(tfidf[1], qvec)
    

#%%
filt =  np.where(tfvec[-1]==1)
d2filt = tfvec[1][filt]


