import numpy as np
import os
import time
import string
import random
import sys
import SPEC

# description : quest. location. previous action. mislead (not including quest)
# 4 sentences / 20 words

'''
description = {}
agent_location = None
agent_quest = None
agent_mislead = None
quest_checklist = False
quest_completed_count = 0
#home_quest = you are hungry --> room = kitchen, action = eat apple
#home_quest = you are sleepy --> room = bedroom, action = sleep bed
#home_quest = you are bored --> room = livingroom, action = watch tv
#home_quest = you are getting fat --> room = garden, action = exercise body
#home_quest = you are dirty --> room = toilet, action = wash face
'''

class HomeWorld():

    def __init__(self, num_home_rooms=5, seq_length=20, max_step=5):
        self.regions = SPEC.regions

        self.home_rooms = SPEC.home_rooms
        self.school_rooms = SPEC.school_rooms
    
        self.home_locations = SPEC.home_locations
        self.school_locations = SPEC.school_locations

        self.home_actions = SPEC.home_actions
        self.school_actions =  SPEC.school_actions

        self.home_quests = SPEC.home_quests
        self.school_quests = SPEC.school_quests

        self.home_quests_mislead = SPEC.home_quests_mislead
        self.school_quests_mislead = SPEC.school_quests_mislead

        self.all_actions = SPEC.all_actions
        self.objects = SPEC.objects

        self.current_quest = self.home_quests[0]
        self.current_location = self.home_locations[0]
        self.current_quest_mislead = self.home_quests_mislead[0]

        self.time = 0
        self.T = SPEC.T
        self.reward = 0
        self.finish = 0
 
        self.agent_pre_action = self.home_actions[0]
        self.agent_pre_object = self.objects[0]
        self.agent_action = self.home_actions[0]
        self.agent_object = self.objects[0]

    def get_random_location(self):
        return random.choice(self.home_locations)


    def get_random_quest(self):
        return random.choice(self.home_quests)


    def get_random_quests_mislead(self):
        return random.choice(self.home_quests_mislead)


    def new_game(self):
        self.current_quest = self.get_random_quest()
        self.current_location = self.get_random_location()
        self.time = 0

        #print ('The new game is now started')
        #print ('Welcome to our Home World!!')
        #print ('Here is your quest: '+self.current_quest)

        return self.current_quest,self.current_location
       
  
    def location_transition(self):
        if self.current_location == self.home_locations[0]: # livingroom
           if self.agent_action == 'go' and self.agent_object == 'east':
              self.current_location = self.home_locations[1] # garden
           if self.agent_action == 'go' and self.agent_object == 'south':
              self.current_location = self.home_locations[3] # bedroom 

        if self.current_location == self.home_locations[1]: # garden
           if self.agent_action == 'go' and self.agent_object == 'east':
              self.current_location = self.home_locations[2] # kitchen
           if self.agent_action == 'go' and self.agent_object == 'west':
              self.current_location = self.home_locations[0] # livingroom 
           if self.agent_action == 'go' and self.agent_object == 'south':
              self.current_location = self.home_locations[4] # toilet

        if self.current_location == self.home_locations[2]: # kitchen
           if self.agent_action == 'go' and self.agent_object == 'west':
              self.current_location = self.home_locations[1] # garden

        if self.current_location == self.home_locations[3]: # bedroom
           if self.agent_action == 'go' and self.agent_object == 'north':
              self.current_location = self.home_locations[0] # livingroom
           if self.agent_action == 'go' and self.agent_object == 'east':
              self.current_location = self.home_locations[4] # toilet

        if self.current_location == self.home_locations[4]: # toilet
           if self.agent_action == 'go' and self.agent_object == 'north':
              self.current_location = self.home_locations[1] # garden
           if self.agent_action == 'go' and self.agent_object == 'west':
              self.current_location = self.home_locations[3] # bedroom  

    def show_pre_action(self):
        return 'You'+' '+str(self.agent_pre_action)+' '+str(self.agent_pre_object)

    def quest_mislead_selection(self):
        if self.current_quest == self.home_quests[0]:
           s = random.choice([1,2,3,4])
           return self.home_quests_mislead[s]

        if self.current_quest == self.home_quests[1]:
           s = random.choice([0,2,3,4])
           return self.home_quests_mislead[s]

        if self.current_quest == self.home_quests[2]:
           s = random.choice([1,0,3,4])
           return self.home_quests_mislead[s]

        if self.current_quest == self.home_quests[3]:
           s = random.choice([1,2,0,4])
           return self.home_quests_mislead[s]

        if self.current_quest == self.home_quests[4]:
           s = random.choice([1,2,3,0])
           return self.home_quests_mislead[s]


    def reward_function(self):
        if self.current_quest == self.home_quests[0]:
             if self.current_location ==  self.home_locations[2] and self.agent_action == 'eat' and self.objects == 'apple': 
                self.reward = 3
             else:
                self.reward = -0.01

        elif self.current_quest == self.home_quests[1]:
             if self.current_location ==  self.home_locations[3] and self.agent_action == 'sleep' and self.objects == 'bed': 
                self.reward = 3
             else:
                self.reward = -0.01
 
        elif self.current_quest == self.home_quests[2]:
             if self.current_location ==  self.home_locations[0] and self.agent_action == 'watch' and self.objects == 'TV': 
                self.reward = 3
             else:
                self.reward = -0.01

        elif self.current_quest == self.home_quests[3]:  
             if self.current_location ==  self.home_locations[1] and self.agent_action == 'exercise' and self.objects == 'body': 
                self.reward = 3
             else:
                self.reward = -0.01
       
        elif self.current_quest == self.home_quests[4]:  
             if self.current_location == self.home_locations[4] and self.agent_action == 'wash' and self.objects == 'body': 
                self.reward = 3
             else:
                self.reward = -0.01

       

    def get_state_reward(self,act,obj): 
        
        self.time += 1
        #self.check_finish()
        quest_output = self.current_quest + ';'
        location_output = self.current_location + ';'

        pre_action_output = self.show_pre_action() + ';'
        self.agent_pre_action = self.agent_action
        self.agent_pre_object = self.agent_object

        quest_mislead_output = self.quest_mislead_selection() + ';'

        # Interaction 
        self.agent_action = act
        self.agent_object = obj

        self.location_transition() 
        self.reward_function()
        reward_output = 'Reward='+str(self.reward)
       
        return quest_output + location_output + pre_action_output + quest_mislead_output + reward_output
        
