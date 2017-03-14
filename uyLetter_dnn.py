import sys
import math
import numpy as np
import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets import base

#from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet

from mnist import DataSet

#load features
print('loading prosody features from text files.....')

train_feats=np.loadtxt('train_feat.txt',dtype='float32')
train_labels=np.loadtxt('train_label.txt',dtype='float32')

test_feats=np.loadtxt('test_feat.txt',dtype='float32')
test_labels=np.loadtxt('test_label.txt',dtype='float32')

valid_feats=np.loadtxt('validation_feat.txt',dtype='float32')
valid_labels=np.loadtxt('validation_label.txt',dtype='float32')


print('finished loading prodoy features and labels from text files')

#prepare dataset for train, test and validation
train = DataSet(train_feats, train_labels, dtype='float32', reshape=False)

test = DataSet(test_feats, test_labels, dtype='float32', reshape=False)

valid = DataSet(valid_feats, valid_labels, dtype='float32', reshape=False)

mProsody=base.Datasets(train=train, validation=valid, test=test)


#print mProsody.train.images.shape

#sys.exit()

#placeholder for input features
X=tf.placeholder(tf.float32,[None,784])

#placeholder for dropout
pkeep=tf.placeholder(tf.float32)

#placeholder for learning rate
lrate=tf.placeholder(tf.float32)

#variables for network parameters
#W=tf.Variable(tf.truncated_normal([15,2],stddev=0.1))
#b=tf.Variable(tf.ones([2])/10)

W1=tf.Variable(tf.truncated_normal([784,512],stddev=0.1))
B1=tf.Variable(tf.ones([512])/10)

W2=tf.Variable(tf.truncated_normal([512,256],stddev=0.1))
B2=tf.Variable(tf.ones([256])/10)

W3=tf.Variable(tf.truncated_normal([256,200],stddev=0.1))
B3=tf.Variable(tf.ones([200])/10)

W4=tf.Variable(tf.truncated_normal([200,128],stddev=0.1))
B4=tf.Variable(tf.ones([128])/10)

#W5=tf.Variable(tf.truncated_normal([30,128],stddev=0.1))
#B5=tf.Variable(tf.ones([128])/10)

#placeholder for correct labels
lab_corr=tf.placeholder(tf.float32,[None,128])

#model

Y1=tf.nn.relu(tf.matmul(X,W1)+B1)

Y1d=tf.nn.dropout(Y1,pkeep)

Y2=tf.nn.relu(tf.matmul(Y1d,W2)+B2)

Y2d=tf.nn.dropout(Y2,pkeep)

Y3=tf.nn.relu(tf.matmul(Y2d,W3)+B3)

Y3d=tf.nn.dropout(Y3,pkeep)

Ylogits=tf.matmul(Y3d,W4)+B4

#Y4=tf.nn.relu(tf.matmul(Y3d,W4)+B4)

#Y4d=tf.nn.dropout(Y4,pkeep)

#Ylogits=tf.matmul(Y4d,W5)+B5

lab_pred=tf.nn.softmax(Ylogits)

#loss function with logit
cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits,labels=lab_corr)
cross_entropy=tf.reduce_mean(cross_entropy)*100


#loss function
#lab_pred=tf.nn.softmax(tf.matmul(Y4d,W5)+B5)
#cross_entropy=-tf.reduce_sum(lab_corr*tf.log(lab_pred))



#accuracy
is_correct=tf.equal(tf.argmax(lab_pred,1),tf.argmax(lab_corr,1))
accuracy=tf.reduce_mean(tf.cast(is_correct,tf.float32))

#optimizer
#optimizer=tf.train.GradientDescentOptimizer(0.003)
optimizer=tf.train.AdamOptimizer(lrate)
train_step=optimizer.minimize(cross_entropy)

#initialize all variable
init=tf.global_variables_initializer()

#create tensorflow session
sess=tf.Session()
sess.run(init)

# learning rate decay
max_learning_rate = 0.003
min_learning_rate = 0.0001
decay_speed = 2000.0

for i in range(5000):
   #load batch of feats and correct labels
   batch_feat,batch_lab=mProsody.train.next_batch(200)
   learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed) 

   train_data={X:batch_feat, lab_corr:batch_lab,lrate:learning_rate,pkeep:1.0}

   sess.run(train_step,feed_dict=train_data)

   acc,cross=sess.run([accuracy,cross_entropy],feed_dict=train_data)
 
   if i%100 == 0:
       print("step %d--->learning rate:%g, training accuracy:%g, cross entropy:%g"%(i,learning_rate,acc,cross))


test_data={X:mProsody.test.images,lab_corr:mProsody.test.labels,pkeep:1.0}

acc,cross=sess.run([accuracy,cross_entropy],feed_dict=test_data)

print("test accuracy:%g"%(acc))

