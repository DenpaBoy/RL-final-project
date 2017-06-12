import numpy

T = 50

# Dictionay all words 
all_words = ['You','are','not','at','in','the','hungry','sleepy','bored','getting', \
             'fat','dirty','going','to','books','class','home','school','living_room','garden', \
             'kitchen','bedroom','toilet','physics','math','music','canteen','field','library','eat', \
             'sleep','watch','go', 'study','wash','borrow','attend','north','south','east', \
             'west','at','in','apple',"body","face","TV","classroom",'exercise','bed']

vocabulary = len(all_words)

# Vector data dimension
vec_dim = vocabulary + 2
seq_len = 20
seq_num = 5  

# (1) Quest (2) Location (3) Pre-Action (4) Quest-mislead (5) Reward
# Ex : 'You are hungry.You are in the kitchen.You eat a apple.You are hungry.Reward=5'
# Reward=5 等號兩邊不能有空格

# "Life of student" setting
regions = ["home", "school"]

home_rooms = ["living_room", "garden", "kitchen", "bedroom", "toilet"]
school_rooms = ["physics", "math", "music", "canteen", "field", "library"]

home_actions = ['go','eat','sleep','watch','exercise']
school_actions =  ['go',"study",'wash', "borrow", "attend"]

home_quests = ["You are hungry", "You are sleepy", "You are bored", "You are getting fat", "You are dirty"]
school_quests = ["You are going to school", "You are going to home",
                 "You are going to borrow books",
                 "You are going to attend math class", "You are going to attend physics class",
                 "You are going to attend music class"]

home_quests_mislead = ["You are not hungry", "You are not sleepy", "You are not bored","You are not getting fat",
                       "You are not dirty"]
school_quests_mislead = ["You are not going to school", "You are not going to home",
                         "You are not going to borrow books",
                         "You are not going to attend music class"]

home_locations = ["You are in the living_room","You are in the garden","You are in the kitchen","You are in the bedroom","You are in the toilet"]
school_locations = ["You are in the phyiscs classroom","You are in the math classroom","You are in the music classroom",
                    "You are at the canteen","You are at the field","You are at the field"]


all_actions = home_actions + school_actions 
objects = ["north", "south", "east", "west", "apple", "bed", "body","face","TV"]

Qa_dim = len(all_actions)
Qo_dim = len(objects)






