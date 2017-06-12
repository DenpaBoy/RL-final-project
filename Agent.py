import tensorflow as tf
import numpy as np
import SPEC
import Environment as env
import Process
import random

class agent:
    def __init__ (self):
        # Environment 
        self.home_env = env.HomeWorld()     

        # Network architecture
        self.vec_dim = SPEC.vec_dim
        self.seq_len = SPEC.seq_len
        self.sen_num = SPEC.seq_num-1
        self.Qa_dim = SPEC.Qa_dim
        self.Qo_dim = SPEC.Qo_dim
        self.all_actions = SPEC.all_actions
        self.objects = SPEC.objects
        self.lstm_hidden_width = self.vec_dim - 10
        self.hidden_width1 = 100
        self.hidden_width2 = 50
        self.rep_mem = []

        # Parameter setting
        self.episodes = 100
        self.epsilon = 0.1
        self.learning_rate = 0.1
        self.gamma = 0.9
        self.T = SPEC.T
        self.loss_a_weight = 0.5
        self.loss_o_weight = 0.5

        # Graph I/O
        self.x = tf.placeholder(tf.float32, [self.sen_num,self.seq_len,self.vec_dim])
        self.mask_a = tf.placeholder(tf.float32, [self.Qa_dim])
        self.mask_o = tf.placeholder(tf.float32, [self.Qo_dim])
        self.y_a = tf.placeholder(tf.float32, [self.Qa_dim])
        self.y_o = tf.placeholder(tf.float32, [self.Qo_dim])

        # Q(s,a) and Q(s,o)
        self.Qa = np.zeros(self.Qa_dim) 
        self.Qo = np.zeros(self.Qo_dim) 
        self.Qa_max = 0
        self.Qo_max = 0
        self.Qa_argmax = 0
        self.Qo_argmax = 0
        self.a_select = 0
        self.o_select = 0

        # Sample Q(s,a) and Q(s,o)
        self.sample_Qa = np.zeros(self.Qa_dim) 
        self.sample_Qo = np.zeros(self.Qo_dim) 
        self.sample_Qa_max = 0
        self.sample_Qo_max = 0
        self.sample_Qa_argmax = 0
        self.sample_Qo_argmax = 0
        self.sample_a_select = 0
        self.sample_o_select = 0
        self.msk_a = np.zeros(self.Qa_dim) 
        self.msk_o = np.zeros(self.Qo_dim) 

        self.ya = np.zeros(self.Qa_dim)
        self.yo = np.zeros(self.Qo_dim)

        # Recode data
        self.episodes_reward_sum = []
        self.reward_sum = 0

        self.initializer = tf.random_normal_initializer(stddev=0.1) 

        self.input_form = tf.split(tf.reshape(tf.transpose(self.x, [1, 0, 2]), [-1, self.vec_dim]), self.seq_len)

        with tf.variable_scope("Representation_Generator"):
             self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_hidden_width)
             self.lstm_output, _ = tf.contrib.rnn.static_rnn(self.lstm_cell, self.input_form, dtype=tf.float32)

        with tf.variable_scope("action_scorer"):

             with tf.variable_scope("Linear1"):
                  self.W1 = tf.get_variable("Weight",shape = [self.lstm_hidden_width, self.hidden_width1],initializer = self.initializer)
                  self.b1 = tf.get_variable("Bias",shape = [self.hidden_width1],initializer = self.initializer)
                  self.h1 = tf.nn.bias_add(tf.matmul(self.lstm_output[-1],self.W1),self.b1)

             with tf.variable_scope("Relu2"):
                  self.W2 = tf.get_variable("Weight",shape = [self.hidden_width1, self.hidden_width2],initializer = self.initializer)
                  self.b2 = tf.get_variable("Bias",shape = [self.hidden_width2],initializer = self.initializer)
                  self.h2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(self.h1,self.W2),self.b2))

             with tf.variable_scope("Linear3a"):
                  self.W3a = tf.get_variable("Weight",shape = [self.hidden_width2, self.Qa_dim],initializer = self.initializer)
                  self.b3a = tf.get_variable("Bias",shape = [self.Qa_dim],initializer = self.initializer)
                  self.Q_a = tf.nn.bias_add(tf.matmul(self.h2,self.W3a),self.b3a)

             with tf.variable_scope("Linear3o"):
                  self.W3o = tf.get_variable("Weight",shape = [self.hidden_width2, self.Qo_dim],initializer = self.initializer)
                  self.b3o = tf.get_variable("Bias",shape = [self.Qo_dim],initializer = self.initializer)
                  self.Q_o = tf.nn.bias_add(tf.matmul(self.h2,self.W3o),self.b3o)
                  

    def train(self):

        self.joint_loss = tf.reduce_mean(self.loss_a_weight*tf.squared_difference(tf.multiply(self.Q_a,self.mask_a),self.y_a)) + \
                          tf.reduce_mean(self.loss_o_weight*tf.squared_difference(tf.multiply(self.Q_o,self.mask_o),self.y_o))
                          
        self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.joint_loss)   

        init = tf.global_variables_initializer()  

        with tf.Session() as sess:

             sess.run(init) 

             for e in range(self.episodes):
 
                 self.home_env.new_game()

                 descriptions = self.home_env.get_state_reward(random.choice(self.all_actions),random.choice(self.objects))

                 self.reward_sum = 0

                 for t in range(1,self.T+1):

                     seqs_tensor,_ = Process.seqs_tensor_encoder(descriptions)

                     self.Qa,self.Qo = sess.run([self.Q_a,self.Q_o],feed_dict={self.x:seqs_tensor})
                     self.Qa = np.mean(self.Qa, axis=0)
                     self.Qo = np.mean(self.Qo, axis=0)
                     self.Qa_argmax = np.argmax(self.Qa)
                     self.Qa_max = self.Qa[self.Qa_argmax]

                     self.Qo_argmax = np.argmax(self.Qo)
                     self.Qo_max = self.Qo[self.Qo_argmax]
                               
                     # Epsilon-greedy
                     if np.random.uniform() <= self.epsilon:
                        self.a_select = np.random.randint(self.Qa_dim)
                     else: 
                        self.a_select = self.Qa_argmax
                        

                     if np.random.uniform() <= self.epsilon:
                        self.o_select = np.random.randint(self.Qo_dim)
                     else: 
                        self.o_select = self.Qo_argmax

                     # Get state and reward
                     next_descriptions = self.home_env.get_state_reward(self.all_actions[self.a_select],self.objects[self.o_select])
                     next_seqs_tensor,reward = Process.seqs_tensor_encoder(next_descriptions)
                     descriptions = next_descriptions
                   
                     self.reward_sum += reward 

                     if reward > 0:
                        self.rep_mem.append([seqs_tensor,self.a_select,self.o_select,reward,next_seqs_tensor])
                     
                     '''
                     self.sample_Qa = np.zeros(self.Qa_dim) 
                     self.sample_Qo = np.zeros(self.Qo_dim) 
                     self.sample_Qa_max = 0
                     self.sample_Qo_max = 0
                     self.sample_Qa_argmax = 0
                     self.sample_Qo_argmax = 0
                     self.sample_a_select = 0
                     self.sample_o_select = 0

                     self.x = tf.placeholder(tf.float32, [self.sen_num,self.seq_len,self.vec_dim])
                     self.mask_a = tf.placeholder(tf.float32, [self.Qa_dim])
                     self.mask_o = tf.placeholder(tf.float32, [self.Qo_dim])
                     self.y_a = tf.placeholder(tf.float32, [self.Qa_dim])
                     self.y_o = tf.placeholder(tf.float32, [self.Qo_dim])
                     '''

                     # Start update network
                     if len(self.rep_mem) >= 1:

                        [sample_seqs_tensor,sample_a,sameple_o,sample_r,sample_next_seq_tensor] = self.rep_mem[np.random.randint(len(self.rep_mem))]
                        self.sample_Qa,self.sample_Qo = sess.run([self.Q_a,self.Q_o],feed_dict={self.x:sample_next_seq_tensor})
                        self.sample_Qa = np.mean(self.sample_Qa, axis=0)
                        self.sample_Qo = np.mean(self.sample_Qo, axis=0)
 
                        self.sample_Qa_argmax = np.argmax(self.sample_Qa)
                        self.sample_Qa_max = self.Qa[self.sample_Qa_argmax]

                        self.sample_Qo_argmax = np.argmax(self.sample_Qo)
                        self.sample_Qo_max = self.Qo[self.sample_Qo_argmax]   
                     
                        self.ya[self.sample_Qa_argmax] = sample_r + self.gamma*self.sample_Qa_max
                        self.yo[self.sample_Qo_argmax] = sample_r + self.gamma*self.sample_Qo_max
                        
                        self.msk_a[sample_a] = 1
                        self.msk_o[sample_o] = 1
 
                        _ = sess.run([opt],feed_dict={self.x:sample_seqs_tensor,
                                                      self.mask_a:self.msk_a,
                                                      self.mask_o:self.msk_o,
                                                      self.y_a:self.ya,
                                                      self.y_o:self.yo})
                 
                 print('Episode '+str(e)+' : Reward sum ='+str(self.reward_sum))
                 self.episodes_reward_sum.append(self.reward_sum)
                   

                 

                        

