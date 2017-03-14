import sys
import math
import numpy as np
import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets import base

from mnist import DataSet

#load features
print('loading features from text files.....')

train_feats=np.loadtxt('train_feat.txt',dtype='float32')

train_feats=np.reshape(train_feats,[-1, 28, 28,1],order='C')

train_labels=np.loadtxt('train_label.txt',dtype='float32')

test_feats=np.loadtxt('test_feat.txt',dtype='float32')

test_feats=np.reshape(test_feats,[-1, 28, 28,1],order='C')

test_labels=np.loadtxt('test_label.txt',dtype='float32')

valid_feats=np.loadtxt('validation_feat.txt',dtype='float32')

valid_feats=np.reshape(valid_feats,[-1, 28, 28,1],order='C')

valid_labels=np.loadtxt('validation_label.txt',dtype='float32')


print('finished loading features and labels from text files')


#prepare dataset for train, test and validation
train = DataSet(train_feats, train_labels, dtype='float32', reshape=False)

test = DataSet(test_feats, test_labels, dtype='float32', reshape=False)

valid = DataSet(valid_feats, valid_labels, dtype='float32', reshape=False)

uyLetter=base.Datasets(train=train, validation=valid, test=test)

#print uyLetter.train.images.shape

#placeholder for input feature
X=tf.placeholder(tf.float32,[None,28,28,1])

#place holder for correct label
Y_correct=tf.placeholder(tf.float32,[None,128])

lrate=tf.placeholder(tf.float32)

pkeep=tf.placeholder(tf.float32)

#three convolutional layer

f1=12
f2=24
f3=48

full=400

W1=tf.Variable(tf.truncated_normal([6, 6, 1, f1],stddev=0.1))
B1=tf.Variable(tf.ones([f1])/10)

W2=tf.Variable(tf.truncated_normal([5, 5, f1, f2],stddev=0.1))
B2=tf.Variable(tf.ones([f2])/10)

W3=tf.Variable(tf.truncated_normal([4, 4, f2, f3],stddev=0.1))
B3=tf.Variable(tf.ones([f3])/10)


#full connect layer
W4=tf.Variable(tf.truncated_normal([7*7*f3,full],stddev=0.1))
B4=tf.Variable(tf.ones([full])/10)

W5=tf.Variable(tf.truncated_normal([full,128],stddev=0.1))
B5=tf.Variable(tf.ones([128])/10)


#model for convolutional layers
stride=1
Y1=tf.nn.relu(tf.nn.conv2d(X,W1,strides=[1,stride,stride,1],padding='SAME')+B1)

stride=2
Y2=tf.nn.relu(tf.nn.conv2d(Y1,W2,strides=[1,stride,stride,1],padding='SAME')+B2)

stride=2
Y3=tf.nn.relu(tf.nn.conv2d(Y2,W3,strides=[1,stride,stride,1],padding='SAME')+B3)


#model for full connect layer

Y3_new=tf.reshape(Y3,[-1, 7*7*f3])
Y4=tf.nn.relu(tf.matmul(Y3_new,W4)+B4)
Y4d=tf.nn.dropout(Y4,pkeep)
Ylogits=tf.matmul(Y4d,W5)+B5
Y=tf.nn.softmax(Ylogits)

#cross entropy
cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits,labels=Y_correct)
cross_entropy=tf.reduce_mean(cross_entropy)*100

#accuracy
is_correct=tf.equal(tf.argmax(Y,1),tf.argmax(Y_correct,1))
accuracy=tf.reduce_mean(tf.cast(is_correct,tf.float32))


#training step
train_step=tf.train.AdamOptimizer(lrate).minimize(cross_entropy)


#initialize all variable
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

for i in range(10000):
   batch_X,batch_Y=uyLetter.train.next_batch(100)

   # learning rate decay
   max_learning_rate = 0.003
   min_learning_rate = 0.00001
   decay_speed = 3000.0
   learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)

   train_data={X:batch_X,Y_correct:batch_Y,lrate:learning_rate,pkeep:0.5}

   sess.run(train_step,feed_dict=train_data)

   acc,cross=sess.run([accuracy,cross_entropy],feed_dict=train_data)

   if i%100 == 0:
      print("step %d--->training accuracy: %g\tcross entropy: %g"%(i,acc,cross))


test_data={X:uyLetter.test.images, Y_correct:uyLetter.test.labels,pkeep:1.0}

acc,cross=sess.run([accuracy,cross_entropy],feed_dict=test_data)

print("accuracy of test on Uyghur letter: %g"%(acc))

