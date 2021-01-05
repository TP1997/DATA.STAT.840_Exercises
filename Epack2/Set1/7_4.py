import hmmlearn, hmmlearn.hmm
import numpy as np

#%%
root = '/home/tuomas/Python/DATA.STAT.840/Epack2/Set1/'

#%% Load the data
sentences = []
with open(root+'hmm_sentences.txt', 'r') as f:
    line = True
    while(line):
        line = f.readline()
        sentences.append(line)
        
    f.close()

sentences = [s[:-1] for s in sentences[:-1]]

#%% Create unique vocabularities
sentences_splitted = []
for s in sentences:
    sentences_splitted.append(s.split())

vocabularies=[]
indices_in_vocabularies=[]
for s in sentences_splitted:
    uniqueresults = np.unique(s, return_inverse=True)
    vocabularies.append(uniqueresults[0])
    indices_in_vocabularies.append(uniqueresults[1])
    
#%% Create an unified vocabulary
# Concatenate all vocabularies
tempvocabulary = []
for i in range(len(sentences_splitted)):
    tempvocabulary.extend(vocabularies[i])
    
# Find the unique elements among all vocabularies
uniqueresults = np.unique(tempvocabulary,return_inverse=True)
unifiedvocabulary = uniqueresults[0]
wordindices = uniqueresults[1]

# Translate previous indices to the unified vocabulary.
vocabularystart = 0
indices_in_unifiedvocabulary = []
for i in range(len(sentences_splitted)):
    # In order to shift word indices, we must temporarily
    # change their data type to a Numpy array
    tempindices = np.array(indices_in_vocabularies[i])
    tempindices = tempindices + vocabularystart
    tempindices = wordindices[tempindices]
    indices_in_unifiedvocabulary.append(tempindices)
    vocabularystart += len(vocabularies[i])

#%% Transform data for hmm
concatenated_data = []
documentlengths = []
for i in range(len(indices_in_unifiedvocabulary)):
    concatenated_data.extend(indices_in_unifiedvocabulary[i])
    documentlengths.append(len(indices_in_unifiedvocabulary[i]))
concatenated_data=np.matrix(concatenated_data).T  

#%% Fit the model
hmm = hmmlearn.hmm.MultinomialHMM(n_components=5, n_iter=100, verbose=True)
hmm_fitted = hmm.fit(concatenated_data, lengths=documentlengths)

#%% Inspect start, transition, and emission probabilities
from pandas import DataFrame
print('\nStart probabilities:\n')
print(hmm_fitted.startprob_)
print('\nEmission probabilities:\n')
print(DataFrame(hmm_fitted.emissionprob_))
print('\nTransition probabilities:\n')
print(DataFrame(hmm_fitted.transmat_))

#%%
topwordidx1 = np.argmax(hmm_fitted.emissionprob_[4])
print(unifiedvocabulary[topwordidx1])





















