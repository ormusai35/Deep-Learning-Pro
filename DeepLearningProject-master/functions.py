import tensorflow as tf
import numpy as np
import nltk
from nltk.tokenize import word_tokenize




#that function took filePath 
#and return nltk split vector of words
def file2Words(file):
	return word_tokenize(open(file,'r').read())
	
def	words2WordsAndPOS(words):
	return nltk.pos_tag(words)
	
#that function took vector of words and return 3-gram	
def wordsTo3grams(wordsAndPOS):
	words = [i for (i,j) in wordsAndPOS]
	POS = [j for (i,j) in wordsAndPOS]
	return list(zip(list(nltk.ngrams(words,3)),list(nltk.ngrams(POS,3))))

def wordsAndPosTo3grams(wordsAndPOS):
	return list(nltk.ngrams(wordsAndPOS,3))

def gramToFlag(grams):
	vector=[]
	for gram in grams[:len(grams)-1]:
		if grams[grams.index(gram)+1][2][0] in ['.','?',',',':','!']:
			vector.append((gram,1))
		else:
			vector.append((gram,0))
	return vector

def prepare_vocabulary(grams):
    idx = 0
    gram2location={}
    for gram in grams:
	    if gram not in gram2location:
		    gram2location[gram]=idx
		    idx += 1      
    return gram2location

def convert2vec(gram,gram2location):
	res_vec = np.zeros(len(gram2location))
	if gram in gram2location:
		res_vec[gram2location[gram]] += 1
	return res_vec
	
def logistic_fun(z):
    return 1/(1.0 + np.exp(-z))





 




