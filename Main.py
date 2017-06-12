import tensorflow as tf
import numpy as np
import Environment
import Process
import SPEC
import Agent

with tf.variable_scope("Train"):
     agent = Agent.agent()
     agent.train()

'''
env = Environment.HomeWorld()

env.new_game()

descriptions = env.get_state_reward('eat','apple')



tensor,reward = Process.seqs_tensor_encoder(descriptions)
act,obj = Process.act_obj_str_decoder(0,0)


print(descriptions)
print(tensor)
print(reward)
print(act)
print(obj)
'''
