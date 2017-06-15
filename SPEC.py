import numpy

T = 50

# Dictionay all words 
all_words = ["living_room", "garden", "kitchen", "bathroom", "bedroom", "go", "west", "east", "south", "north", \
             "eat", "apple", "sleep", "bed", "watch", 'wash', "TV", "exercise", "body", "You", "are", "hungry", "sleepy", \
             "bored", "dirty", "getting", "fat",'the','in']


vocabulary = len(all_words)

# Vector data dimension
vec_dim = vocabulary + 2
seq_len = 10
seq_num = 2 
des_len = seq_len*seq_num
step_reward = 1
step_negative_reward = -0.01
step_penalty = -0.5

# (1) Quest (2) Location (3) Pre-Action (4) Quest-mislead (5) Reward
# Ex : 'You are hungry.You are in the kitchen.You eat a apple.You are hungry.Reward=5'
# Reward=5 等號兩邊不能有空格

# "Life of student" setting
home_quests = ["You are bored", 'You are getting fat', "You are hungry", "You are sleepy", "You are dirty"]

home_locations = ["You are in the living_room","You are in the garden","You are in the kitchen","You are in the bedroom","You are in the bathroom"]

home_actions = ['go west','go east','go south','go north','watch TV','exercise body','eat apple','sleep bed','wash body']

Qa_dim = len(home_actions)






