import numpy as np
import SPEC

# Parameter
global seq_num
global seq_len
global vec_dim
global Qa_dim

seq_num = SPEC.seq_num
seq_len = SPEC.seq_len
vec_dim = SPEC.vec_dim
des_len = SPEC.des_len

Qa_dim = SPEC.Qa_dim

vocabulary = SPEC.vocabulary


# All words
all_words = SPEC.all_words

# Tuple list 
id_list = list(range(1,vocabulary+1))   # 0: <SOS> , end: <EOS>             

# Word to encoding tuple table 
global table
table = dict(zip(all_words,id_list)) 


# Spilt string to list
def seqs_str_to_lists(seqs_str):

    quest_str,location_str = seqs_str.split(';')

    quest = quest_str.split(' ')
    location = location_str.split(' ')

    return quest,location


# String sequences to tensor
def seqs_tensor_encoder(seqs_str):

    seq_tensor = np.zeros([des_len,vec_dim])
    quest,location = seqs_str_to_lists(seqs_str)   
      
    for w in range(1,1+len(quest)):
        seq_tensor[0+w,table[quest[w-1]]] = 1.
        
    for w in range(1,1+len(location)):
        seq_tensor[seq_len+w,table[location[w-1]]] = 1.


    return seq_tensor



