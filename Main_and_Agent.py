import tensorflow as tf
import numpy as np
import Environment as env
import Process
import SPEC
import random
import collections
import matplotlib.patches as mpatches             
import matplotlib.pyplot as plt  
import os

##### Environment #########
home_env = env.HomeWorld()     

##### Network architecture #####
vec_dim = SPEC.vec_dim
seq_len = SPEC.seq_len
sen_num = SPEC.seq_num

des_len = SPEC.des_len

Qa_dim = SPEC.Qa_dim
actions = SPEC.home_actions

lstm_hidden_width = vec_dim
hidden_width1 = 25
hidden_width2 = 20
mem_max_len = 500
p_rep_mem = collections.deque()
r_rep_mem = collections.deque()

##### Parameter setting #####
episodes = 200
display_step = 10
epsilon = 1.
learning_rate = 0.002
gamma = 0.6
fraction = 0.25
T = SPEC.T
loss_a_weight = 1



##### Graph I/O ##################################
x = tf.placeholder(tf.float32, [des_len,vec_dim])
y_a = tf.placeholder(tf.float32)
y_o = tf.placeholder(tf.float32)
sample_a_select = tf.placeholder(tf.int32)
sample_o_select = tf.placeholder(tf.int32)

##### Q(s,a) ##### 
Qa_max = 0
Qa_argmax = 0
a_select = 0

##### Sample Q(s,a) #######
sample_next_Qa_max = 0
sample_next_Qa_argmax = 0

ya = np.zeros(Qa_dim)

##### Recode data ###################
episodes_accumulative_reward_sum = []
episodes_reward_sum = []
demo_data = []
e_x = np.arange(1,episodes+1) 

trash = None

initializer = tf.random_normal_initializer(stddev=0.1) 

##### Construct network ######################################################################################
input_form = tf.split(x, des_len)

with tf.variable_scope("Representation_Generator"):
     lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_hidden_width)
     lstm_output, _ = tf.contrib.rnn.static_rnn(lstm_cell, input_form, dtype=tf.float32)
     mean_pool = lstm_output[0]
     for w in range(1,des_len): 
         mean_pool += lstm_output[w]
     mean_pool /= des_len

with tf.variable_scope("action_scorer"):

     with tf.variable_scope("Linear1"):
          W1 = tf.get_variable("Weight",shape = [lstm_hidden_width, hidden_width1],initializer = initializer)
          b1 = tf.get_variable("Bias",shape = [hidden_width1],initializer = initializer)
          h1 = tf.nn.bias_add(tf.matmul(mean_pool,W1),b1)

     with tf.variable_scope("Relu2"):
          W2 = tf.get_variable("Weight",shape = [hidden_width1, hidden_width2],initializer = initializer)
          b2 = tf.get_variable("Bias",shape = [hidden_width2],initializer = initializer)
          h2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(h1,W2),b2))

     with tf.variable_scope("Linear3a"):
          W3a = tf.get_variable("Weight",shape = [hidden_width2, Qa_dim],initializer = initializer)
          b3a = tf.get_variable("Bias",shape = [Qa_dim],initializer = initializer)
          Q_a = tf.nn.bias_add(tf.matmul(h2,W3a),b3a)


     Q_a_argmax = tf.argmax(Q_a,axis=1)    
     sample_Qa = Q_a[0,sample_a_select]



##### Training setting ######################################################
loss_a = tf.reduce_mean(tf.squared_difference(sample_Qa,y_a))
opt_a = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_a) 

init = tf.global_variables_initializer()  
saver = tf.train.Saver()

##### Testing setting #######
test_round = 100
success_num = 0


##### Training and testing ############################
with tf.Session() as sess:

     sess.run(init)    
     accumulative_reward_sum = 0

     for e in range(episodes):
 
         home_env.new_game()
         home_env.current_quest = home_env.quests[0]

         descriptions,_ = home_env.get_state_reward(random.choice(actions))
         reward_sum = 0    
        
         if epsilon > 0.1:
            epsilon -= 0.01

         for t in range(1,T+1):

             seqs_tensor = Process.seqs_tensor_encoder(descriptions)

             Qa,Qa_argmax = sess.run([Q_a,Q_a_argmax],feed_dict={x:seqs_tensor})

             ### Epsilon-greedy ###
             if np.random.uniform() <= epsilon:
                a_select = np.random.randint(Qa_dim)
             else: 
                a_select = int(Qa_argmax)
  

             ### Get state and reward ###
             next_descriptions,reward = home_env.get_state_reward(actions[a_select])
             next_seqs_tensor = Process.seqs_tensor_encoder(next_descriptions)
             
   
             reward_sum += reward 

             ### Store replay memory ###
             if reward > 0:
                p_rep_mem.append([seqs_tensor,a_select,reward,next_seqs_tensor])
             else:
                r_rep_mem.append([seqs_tensor,a_select,reward,next_seqs_tensor])
             
             descriptions = next_descriptions

             ### Start update network ###
             if len(p_rep_mem) >= 1 and len(r_rep_mem) >= 1 :
                
                if np.random.uniform() <= fraction :
                   [sample_seqs_tensor,sample_a,sample_r,sample_next_seqs_tensor] = p_rep_mem[np.random.randint(len(p_rep_mem))]
                else :
                   [sample_seqs_tensor,sample_a,sample_r,sample_next_seqs_tensor] = r_rep_mem[np.random.randint(len(r_rep_mem))]   

                sample_next_Qa,sample_next_Qa_argmax = sess.run([Q_a,Q_a_argmax],feed_dict={x:sample_next_seqs_tensor})

                sample_next_Qa_max = sample_next_Qa[0,sample_next_Qa_argmax]

                ya = sample_r + gamma*sample_next_Qa_max
                
                Qa = sess.run([Q_a],feed_dict={x:seqs_tensor}) 

 
                _,temp_loss_a = sess.run([opt_a,loss_a],feed_dict={y_a:ya,x:sample_seqs_tensor,sample_a_select:sample_a})
              
             ### Pop replay memory
             if len(p_rep_mem) > mem_max_len:
                trash = p_rep_mem.popleft()

             if len(r_rep_mem) > mem_max_len:
                trash = r_rep_mem.popleft()

         accumulative_reward_sum += reward_sum
      
         if (e+1)%display_step == 0:
            print('Episode '+str(e+1)+' : Reward sum = '+str(reward_sum)+' , accumulative reward sum = '+str(accumulative_reward_sum))
         
         episodes_accumulative_reward_sum.append(accumulative_reward_sum)
         episodes_reward_sum.append(reward_sum)

         
     # demo_data
     ### Testing : complete ###
     success_num = 0 
     home_env.new_game()
     for test in range(1,test_round+1):
         
         home_env.current_quest = home_env.quests[0]
         descriptions,_ = home_env.get_state_reward(random.choice(actions))
         
         for t in range(1,T+1):

             seqs_tensor = Process.seqs_tensor_encoder(descriptions)
             _,Qa_argmax = sess.run([Q_a,Q_a_argmax],feed_dict={x:seqs_tensor})
 
             quest_str,location_str = descriptions.split(';')

             ### Epsilon-greedy ###
             if np.random.uniform() <= epsilon:
                a_select = np.random.randint(Qa_dim)
             else: 
                a_select = int(Qa_argmax)

             demo_data.append(quest_str+'. '+location_str+'. '+'You '+home_env.actions[a_select]+'. \n')

             if quest_str == home_env.quests[0] and location_str == home_env.locations[0] and a_select == 4 :
                success_num += 1
                demo_data.append('Success !!. \n')
                break
                 
             descriptions,_ = home_env.get_state_reward(actions[a_select])
         
         success_rate = float(success_num/test_round)

     print('Success rate : '+str(success_rate))         

    

     plt.figure(1)                                
     plt.xlabel('Episode')                                         
     plt.ylabel('Reward sum')                    
     plt.title('Agent accumulative reward sum')          
     plt.plot(e_x,episodes_accumulative_reward_sum,'b')             
     plt.savefig('Agent accumulative reward sum.png') 

     plt.figure(2)                                
     plt.xlabel('Episode')                                         
     plt.ylabel('Reward sum')                    
     plt.title('Agent reward sum')          
     plt.plot(e_x,episodes_reward_sum,'b')             
     plt.savefig('Agent reward sum.png') 

     f1 = open('Success_rate.txt','w')  
     f1.write('Success_rate = '+str(success_rate*100)+' %')
     f1.close()

     f2 = open('Demo_data.txt','w') 
     for i in range(test_round):
         f2.write(demo_data[i]) 
     f2.close()

 
  

     save_path = saver.save(sess, "/home/andrew/Desktop/New-RL-final-project")
  












