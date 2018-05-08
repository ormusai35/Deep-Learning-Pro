import tensorflow as tf
import numpy as np
import nltk
import pickle
from nltk.tokenize import word_tokenize
from functions import *

f = open('voca.pckl', 'rb')
gram2location = pickle.load(f)
f.close()

vocabulary_size=len(gram2location)
features = vocabulary_size

W = tf.Variable(tf.zeros([features,1]),name='W')
b = tf.Variable(tf.zeros([1]),name='b')

saver = tf.train.Saver()
sess = tf.Session() 
saver.restore(sess,"trainData/W&b.ckpt")

test1 = "in every sense of the term And to speak in all seriousness"
test33 = "i love to eat pizza pasta and meatballs"
test = wordsAndPosTo3grams(words2WordsAndPOS(word_tokenize(test1)))[3]
test2 = wordsAndPosTo3grams(words2WordsAndPOS(word_tokenize(test1)))[7]
test3 = wordsAndPosTo3grams(words2WordsAndPOS(word_tokenize(test1)))[9]
test4 = wordsAndPosTo3grams(words2WordsAndPOS(word_tokenize(test1)))[0]
test5 = wordsAndPosTo3grams(words2WordsAndPOS(word_tokenize(test33)))[0]
test6 = wordsAndPosTo3grams(words2WordsAndPOS(word_tokenize(test33)))[1]
test7 = wordsAndPosTo3grams(words2WordsAndPOS(word_tokenize(test33)))[2]
test8 = wordsAndPosTo3grams(words2WordsAndPOS(word_tokenize(test33)))[3]
test9 = wordsAndPosTo3grams(words2WordsAndPOS(word_tokenize(test33)))[4]
test10 = wordsAndPosTo3grams(words2WordsAndPOS(word_tokenize(test33)))[5]
print(test)
print(test2)
print(test3)
print(test4)
print("##################hey##############")
print(test5)
print(test6)
print(test7)
print(test8)
print(test9)
print(test10)

#print(flags)

# dont count because we use the same data for test
print('Prediction for: "'  '"', logistic_fun(np.matmul(np.array([convert2vec(test,gram2location)]),sess.run(W)) + sess.run(b))[0][0])
print('Prediction for: "'  '"', logistic_fun(np.matmul(np.array([convert2vec(test2,gram2location)]),sess.run(W)) + sess.run(b))[0][0])
print('Prediction for: "'  '"', logistic_fun(np.matmul(np.array([convert2vec(test3,gram2location)]),sess.run(W)) + sess.run(b))[0][0])

print("##################hey##############")

print('Prediction for: "'  '"', logistic_fun(np.matmul(np.array([convert2vec(test4,gram2location)]),sess.run(W)) + sess.run(b))[0][0])
print('Prediction for: "'  '"', logistic_fun(np.matmul(np.array([convert2vec(test5,gram2location)]),sess.run(W)) + sess.run(b))[0][0])
print('Prediction for: "'  '"', logistic_fun(np.matmul(np.array([convert2vec(test6,gram2location)]),sess.run(W)) + sess.run(b))[0][0])
print('Prediction for: "'  '"', logistic_fun(np.matmul(np.array([convert2vec(test7,gram2location)]),sess.run(W)) + sess.run(b))[0][0])
print('Prediction for: "'  '"', logistic_fun(np.matmul(np.array([convert2vec(test8,gram2location)]),sess.run(W)) + sess.run(b))[0][0])
print('Prediction for: "'  '"', logistic_fun(np.matmul(np.array([convert2vec(test9,gram2location)]),sess.run(W)) + sess.run(b))[0][0])
print('Prediction for: "'  '"', logistic_fun(np.matmul(np.array([convert2vec(test10,gram2location)]),sess.run(W)) + sess.run(b))[0][0])




while 1==1:
	1





 




