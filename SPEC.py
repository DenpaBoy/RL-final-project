import numpy

# Dictionay all words 
all_words = ['You','are','not','at','in','the','hungry','sleepy','bored','getting', \
             'fat','dirty','going','to','books','class','home','school','living','garden', \
             'kitchen','bedroom','toilet','physics','math','music','canteen','field','library','eat', \
             'sleep','watch','go', 'study','bath','borrow','attend','north','south','east', \
             'west','at','in','apple']

vocabulary = len(all_words)

# Vector data dimension
vec_dim = vocabulary + 2
seq_len = 20
seq_num = 5  

# (1) Quest (2) Location (3) Pre-Action (4) Action-Result (5) Reward
# Ex : 'You are hungry.You are in the kitchen.You eat a apple.You are hungry.Reward=5'
# Reward=5 等號兩邊不能有空格

# "Life of student" setting
regions = ['home','school']
home_rooms = ['living','garden','kitchen','bedroom','toilet']
school_rooms = ['physics','math','music','canteen','field','library']
actions = ['eat','sleep','watch','go','study','bath','borrow','attend']
objects = ['north','south','east','west']
quests = ['You are hungry', 'You are sleepy', 'You are bored', 'You are getting fat', \
          'You are dirty', 'You are going to school' , 'You are going to home', 'You are going to borrow books', \
          'You are going to attend math class',\
          'You are going to attend physics class',\
          'You are going to attend music class']
quests_mislead = ['You are not hungry', 'You are not sleepy', 'You are not bored', \
                  'You are not getting fat', 'You are not dirty', 'You are not going to school' ,\
                  'You are not going to home', 'You are not going to borrow books', \
                  'You are not going to attend math class', 'You are not going to attend physics class',\
                  'You are not going to attend music class']





