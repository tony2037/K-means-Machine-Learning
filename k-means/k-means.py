import tensorflow as tf
import numpy as np
import time

#help us to graph
import matplotlib
import matplotlib.pyplot as plt

#import datasets we need by scikit-learn
from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets.samples_generator import make_circles
#fuck Here I install scipy a matherical package

#set up data type , here i choose blobs to make it simpler
DATA_TYPE = "blobs"

#Set up Number of clusters in train data , if we choose circle,2 is enough
K = 4
if(DATA_TYPE == "circle"):
    K = 2
else:
    K = 4

#Set up max of iterations , if condition is not met , here I choose 1000
MAX_ITERS = 1000

#To caculate the time we use , record the begining time
start = time.time()

#Since we have chosen four clusters , We have to give four center points for training data
centers = [(-2, -2), (-2, 1.5), (1.5, -2), (2, 1.5)]
#set up the training set
#for blobs:
 #n_samples:number of data,which means we have 200 points
 #centers = centers
 #n_features = dimmension , here we choose plane so = 2
 #cluster_std = std
 #shuffle:if we mix up samples,here I choose false
 #random_state:random seed
#for circles:
 #noise: random noise data set up to the sample set
 #factor: the ratio factor  between circle data set
if(DATA_TYPE == "circle"):
    data, features = make_circles(n_samples=200,shuffle=True,noise=None,factor=0.4)
else:
    data, features = make_blobs(n_samples=200,centers=centers,n_features=2,cluster_std=0.8,shuffle=False,random_state=42)

#Draw the four centers
#.transpose[0]: x   .transpose[1]: y
fig, ax = plt.subplots()
ax.scatter(np.asarray(centers).transpose()[0], np.asarray(centers).transpose()[1], marker = 'o', s = 250)
plt.show()
#Draw the training data
fig, ax = plt.subplots()
if(DATA_TYPE == "blobs"):
    ax.scatter(np.asarray(centers).transpose()[0], np.asarray(centers).transpose()[1], marker = 'o', s = 250)
    ax.scatter(data.transpose()[0],data.transpose()[1], marker = 'o', s = 100 , c = features, cmap =plt.cm.coolwarm)
    plt.plot()
    plt.show()

#Set up tf.Variable
 #points = data
 #cluster_assignments = each points 's cluster
  #for example:
   #cluster_assignments[13]=2 means 13th point belong cluster 2
N = len(data)
points = tf.Variable(data)
cluster_assignments = tf.Variable(tf.zeros([N], dtype=tf.int64))

#centroids: each groups 's centroids
#tf.slice() really fuck up
#random pick 4 point after all
centroids = tf.Variable(tf.slice(points.initialized_value(), [0,0], [K,2]))

sess = tf.Session()
sess.run(tf.initialize_all_variables())

sess.run(centroids)

# Lost function and rep loop
#centroids = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] shape=[4,2]
#tf.tile(centroids, [N, 1]) = [N*[x1,y1], N*[x2,y2], N*[x3,y3], N*[x4,y4]] shape=[4N,2]
#rep_centroids = tf.reshape(tf.tile(centroids, [N,1]), [N,K,2]) = [ [N*[x1,y1]] , [N*[x2,y2]] , [N*[x3,y3]] , [N*[x4,y4]] ]
#The condition of stopping process is : "Centroids stop changing" :: did_assignments_change

rep_centroids = tf.reshape(tf.tile(centroids, [N,1]), [N,K,2])
rep_points = tf.reshape(tf.tile(points, [1, K]),[N, K, 2])
sum_squares = tf.reduce_sum(tf.square(rep_points - rep_centroids), reduction_indices=2)
best_centroids = tf.argmin(sum_squares, 1)
did_assignments_change = tf.reduce_any(tf.not_equal(best_centroids, cluster_assignments))

#total=[[all sum of points of group 1], [all sum of points of group 2], [all sum of points of group 3], [all sum of points of group 4]] shape=[4,2]
#count=[How many points of each group] shape = [4,1]
#total/count = [new centroids] shape = [4,1]
def bucket_mean(data, bucket_ids, num_buckets):
    total = tf.unsorted_segment_sum(data, bucket_ids, num_buckets)
    count = tf.unsorted_segment_sum(tf.ones_like(data), bucket_ids, num_buckets)
    return total/count

means = bucket_mean(points, best_centroids, K)

#Do update
with tf.control_dependencies([did_assignments_change]):
    do_updates = tf.group(centroids.assign(means), cluster_assignments.assign(best_centroids))

changed = True
iters = 0
fig, ax = plt.subplots()
if(DATA_TYPE == "blobs"):
    colourindexes = [2,1,4,3]
else:
    colourindexes = [2,1]

while changed and iters < MAX_ITERS:
    fig, ax = plt.subplots()
    iters +=1
    [changed, _] = sess.run([did_assignments_change, do_updates])
    [centers, assignments] = sess.run([centroids, cluster_assignments])
    ax.scatter(sess.run(points).transpose()[0], sess.run(points).transpose()[1], marker = 'o', s = 200, c = assignments, cmap=plt.cm.coolwarm)
    ax.scatter(centers[:,0], centers[:,1], marker = '^', s = 550, c=colourindexes, cmap=plt.cm.plasma)
    ax.set_title("Iteration " + str(iters))
    plt.savefig("kmeans" + str(iters) + ".png")

ax.scatter(sess.run(points).transpose()[0], sess.run(points).transpose()[1], marker='o', s=200, c=assignments, cmap=plt.cm.coolwarm)
plt.show()
end = time.time()
print("Found in %.2f seconds" %(end-start), iters, "iterations")
print("Centroids: ")
print(centers)
print("Cluster assignment", assignments)