import tensorflow as tf
import numpy as np
import nltk
from nltk.tokenize import word_tokenize

vocabulary_size = 0  #can use "global" keyword
gram2location = {}


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
    for gram in grams:
	    if gram not in gram2location:
		    gram2location[gram]=idx
		    idx += 1      
    return idx

def convert2vec(gram):
    res_vec = np.zeros(vocabulary_size)
    if gram in gram2location:
	    res_vec[gram2location[gram]] += 1
    return res_vec
	
file='lessData.txt'
words=file2Words(file)
wordsAndPOS=words2WordsAndPOS(words)
gram3=wordsAndPosTo3grams(wordsAndPOS)
flags=gramToFlag(gram3)
vocabulary_size=prepare_vocabulary(gram3)

features = vocabulary_size
eps = 1e-12
x = tf.placeholder(tf.float32, [None, features])
y_ = tf.placeholder(tf.float32, [None, 1])
W = tf.Variable(tf.zeros([features,1]))
b = tf.Variable(tf.zeros([1]))
y = tf.nn.sigmoid(tf.matmul(x,W)+b)
loss1 = -(y_ * tf.log(y + eps) + (1 - y_) * tf.log((1 - y) + eps))
loss = tf.reduce_mean(loss1)

update = tf.train.GradientDescentOptimizer(0.3).minimize(loss)

data_x = np.array([convert2vec(gram) for gram in gram3])
data_y = [[i] for (j,i) in flags]+[[0]]

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(0,100000):
	sess.run(update, feed_dict = {x:data_x, y_:data_y}) 
	if i % 10000 == 0 :
		print('Iteration:' , i , ' W:' , sess.run(W)[0] , ' b:' , sess.run(b), ' loss:', loss.eval(session=sess, feed_dict = {x:data_x, y_:data_y}))


saver = tf.train.Saver()		
saver.save(sess,"trainData/W&b.ckpt")

def logistic_fun(z):
    return 1/(1.0 + np.exp(-z))
    
test1 = "in every sense of the term. And to speak in all seriousness,"
test = wordsAndPosTo3grams(words2WordsAndPOS(word_tokenize(test1)))[3]
test2 = wordsAndPosTo3grams(words2WordsAndPOS(word_tokenize(test1)))[7]
test3 = wordsAndPosTo3grams(words2WordsAndPOS(word_tokenize(test1)))[10]
test4 = wordsAndPosTo3grams(words2WordsAndPOS(word_tokenize(test1)))[0]
print(test)
print(test2)
print(test3)
print(test4)
#print(flags)

# dont count because we use the same data for test
print('Prediction for: "'  '"', logistic_fun(np.matmul(np.array([convert2vec(test)]),sess.run(W)) + sess.run(b))[0][0])
print('Prediction for: "'  '"', logistic_fun(np.matmul(np.array([convert2vec(test2)]),sess.run(W)) + sess.run(b))[0][0])
print('Prediction for: "'  '"', logistic_fun(np.matmul(np.array([convert2vec(test3)]),sess.run(W)) + sess.run(b))[0][0])
print('Prediction for: "'  '"', logistic_fun(np.matmul(np.array([convert2vec(test4)]),sess.run(W)) + sess.run(b))[0][0])



#print(len(data_x))

#print(data_y)
#print(len(data_y))
#print([convert2vec(gram) for gram in gram3])
#print (gram2location)
#print(vocabulary_size)
#print (flags)
#print (len(flags))





 




