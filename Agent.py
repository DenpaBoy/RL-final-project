import tensorflow as tf
import numpy as np
import SPEC
import Environment
class Agent:
    def __init__ (self,world):
        self.vec_dim = SPEC.vec_dim
        self.seq_len = SPEC.seq_len
        self.sen_num = SPEC.seq_num-1
        self.hidden_node1 = 200
        self.hidden_node2 = 200
        self.world = world
        self.action_num = len(self.world.actions) 
        self.object_num = len(self.world.objects) 
        initializer = tf.contrib.layers.xavier_initializer()
        #input
        self.w = tf.placeholder(tf.float32, [None, self.seq_len*self.sen_num , self.vec_dim])
        with tf.variable_scope("Representation_Generator"):
            x_s, _ = tf.contrib.rnn.static_rnn(tf.contrib.rnn.BasicLSTMCell(self.vec_dim), tf.unstack(self.w, self.seq_len*self.sen_num, 1), dtype=tf.float32)
            v_s = tf.divide(tf.add_n(x_s),self.seq_len*self.sen_num)
        with tf.variable_scope("action_scorer"):
            with tf.variable_scope("Linear1"):
                W1 = tf.get_variable("Weight",shape = [self.vec_dim, self.hidden_node1],initializer = initializer)
                b1 = tf.get_variable("Bias",shape = [self.hidden_node1],initializer = initializer)
                h1 = tf.matmul(v_s,W1) + b1
            with tf.variable_scope("Relu2"):
                W2 = tf.get_variable("Weight",shape = [self.hidden_node1, self.hidden_node2],initializer = initializer)
                b2 = tf.get_variable("Bias",shape = [self.hidden_node2],initializer = initializer)
                h2 = tf.nn.relu(tf.matmul(h1,W2) + b2)
            with tf.variable_scope("Linear3a"):
                W3a = tf.get_variable("Weight",shape = [self.hidden_node2//2, self.action_num],initializer = initializer)
                b3a = tf.get_variable("Bias",shape = [self.action_num],initializer = initializer)
                self.Qsa = tf.matmul(h2[:,0:self.hidden_node2//2],W3a) + b3a
            with tf.variable_scope("Linear3o"):
                W3o = tf.get_variable("Weight",shape = [self.hidden_node2//2, self.object_num],initializer = initializer)
                b3o = tf.get_variable("Bias",shape = [self.object_num],initializer = initializer)
                self.Qso = tf.matmul(h2[:,self.hidden_node2//2:self.hidden_node2],W3o) + b3o
#Agent(Environment.HomeWorld())
