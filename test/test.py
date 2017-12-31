# refer to the following intro page
# https://www.tensorflow.org/get_started/get_started

from __future__ import print_function

import sys
import tensorflow as tf

print(sys.version)

# Phase-1: Building the computational graph

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
node3 = tf.add(node1, node2)
print(node1, node2)
print("node3:", node3)

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)
add_and_triple = adder_node * 3.


W = tf.Variable([.3], dtype=tf.float32)
h = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x + h

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

# Phase-2: Running the computational graph

sess = tf.Session()
print("sess.run([node1, node2]):", sess.run([node1, node2]))
print("sess.run(node3):", sess.run(node3))

print("sess.run(adder_node, {a: 3, b: 4.5}): ", sess.run(adder_node, {a: 3, b: 4.5}))
print("sess.run(adder_node, {a: [1, 3], b: [2, 4]}): ", sess.run(adder_node, {a: [1, 3], b: [2, 4]}))
print("sess.run(add_and_triple, {a: 3, b: 4.5}): ", sess.run(add_and_triple, {a: 3, b: 4.5}))

# variables are not initialized when you call tf.Variable. 
# To initialize all the variables in a TensorFlow program,
# you must explicitly call a special operation as follows:
init = tf.global_variables_initializer()
sess.run(init)
print("sess.run(linear_model, {x: [1, 2, 3, 4]}): ", sess.run(linear_model, {x: [1, 2, 3, 4]}))
print("sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}): ", sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))


# change the variables then run again
fixW = tf.assign(W, [-1.])
fixh = tf.assign(h, [1.])
sess.run([fixW, fixh])
print("After fix: sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}): ", sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))




