import tensorflow as tf
import numpy as np
import nltk
import pickle
from nltk.tokenize import word_tokenize
from functions import *

vocabulary_size = 0  #can use "global" keyword
gram2location = []

file='lessData.txt'
words=file2Words(file)
wordsAndPOS=words2WordsAndPOS(words)
gram3=wordsAndPosTo3grams(wordsAndPOS)
flags=gramToFlag(gram3)

gram2location = prepare_vocabulary(gram3)
vocabulary_size=len(gram2location)

features = vocabulary_size
eps = 1e-12

x = tf.placeholder(tf.float32, [None, features])
y_ = tf.placeholder(tf.float32, [None, 1])

W = tf.Variable(tf.zeros([features,1]),name='W')
b = tf.Variable(tf.zeros([1]),name='b')
y = tf.nn.sigmoid(tf.matmul(x,W)+b)
loss1 = -(y_ * tf.log(y + eps) + (1 - y_) * tf.log((1 - y) + eps))
loss = tf.reduce_mean(loss1)

update = tf.train.GradientDescentOptimizer(0.3).minimize(loss)

data_x = np.array([convert2vec(gram,gram2location) for gram in gram3])
data_y = [[i] for (j,i) in flags]+[[0]]

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(0,100000):
	sess.run(update, feed_dict = {x:data_x, y_:data_y}) 
	if i % 1000 == 0 :
		print('Iteration:' , i , ' W:' , sess.run(W)[0] , ' b:' , sess.run(b), ' loss:', loss.eval(session=sess, feed_dict = {x:data_x, y_:data_y}))


saver = tf.train.Saver()		
saver.save(sess,"trainData/W&b.ckpt")

f = open('voca.pckl', 'wb')
pickle.dump(gram2location, f)
f.close()
    
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
print('Prediction for: "'  '"', logistic_fun(np.matmul(np.array([convert2vec(test,gram2location)]),sess.run(W)) + sess.run(b))[0][0])
print('Prediction for: "'  '"', logistic_fun(np.matmul(np.array([convert2vec(test2,gram2location)]),sess.run(W)) + sess.run(b))[0][0])
print('Prediction for: "'  '"', logistic_fun(np.matmul(np.array([convert2vec(test3,gram2location)]),sess.run(W)) + sess.run(b))[0][0])
print('Prediction for: "'  '"', logistic_fun(np.matmul(np.array([convert2vec(test4,gram2location)]),sess.run(W)) + sess.run(b))[0][0])









 




