import numpy as np
import SPEC

# Spilt string to list
def seqs_str_to_lists(seqs_str):

    quest_str,location_str,pre_action_str,action_result_str,reward_str = seqs_str.split(';')

    quest = quest_str.split(' ')
    location = location_str.split(' ')
    pre_action = pre_action_str.split(' ')
    action_result = action_result_str.split(' ')

    reward_temp = reward_str.split('=')
    reward = float(reward_temp[1])

    return quest,location,pre_action,action_result,reward

# Parameter
global seq_num
global seq_len
global vec_dim
global Qa_dim
global Qo_dim
global all_actions
global objects


seq_num = SPEC.seq_num
seq_len = SPEC.seq_len
vec_dim = SPEC.vec_dim
Qa_dim = SPEC.Qa_dim
Qo_dim = SPEC.Qo_dim
all_actions = SPEC.all_actions
objects = SPEC.objects

vocabulary = SPEC.vocabulary

# All words
all_words = SPEC.all_words

# Tuple list 
id_list = list(range(1,vocabulary+1))   # 0: <SOS> , end: <EOS>             

# Word to encoding tuple table 
global table
table = dict(zip(all_words,id_list)) 

# String sequences to tensor
def seqs_tensor_encoder(seqs_str):

    seq_tensor = np.zeros([seq_num-1,seq_len,vec_dim])

    # SOS
    seq_tensor[:,0,0] = 1.
    # EOS
    seq_tensor[:,seq_len-1,vec_dim-1] = 1.

    quest,location,pre_action,action_result,reward = seqs_str_to_lists(seqs_str)   
      

    for w in range(1,1+len(quest)):
       
        seq_tensor[0,w,table[quest[w-1]]] = 1.
        
    for w in range(1,1+len(location)):
        seq_tensor[1,w,table[location[w-1]]] = 1.

    for w in range(1,1+len(pre_action)):
        seq_tensor[2,w,table[pre_action[w-1]]] = 1.
        

    for w in range(1,1+len(action_result)):
        seq_tensor[3,w,table[action_result[w-1]]] = 1.

    return seq_tensor,reward


def act_obj_str_decoder(act,obj):
    return all_actions[act],objects[obj]



