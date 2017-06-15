import numpy as np
import os
import time
import string
import random
import sys
import SPEC

# description : quest. location. previous action. mislead (not including quest)
# 4 sentences / 20 words

class HomeWorld():

    def __init__(self, num_rooms=5, seq_length=20, max_step=5):

        self.quests = SPEC.home_quests
        self.locations = SPEC.home_locations
        self.actions = SPEC.home_actions        

        self.current_quest = self.quests[0]
        self.current_location = self.locations[0]
        self.reward = 0

        self.step_negative_reward = SPEC.step_negative_reward
        self.step_reward = SPEC.step_reward
       
        self.agent_action = self.actions[0]

        self.quests_num = len(self.quests)
        self.locations_num = len(self.locations)

    def new_game(self): 
        # self.current_quest = self.quests[np.random.randint(self.quests_num)]
        self.current_quest = self.quests[0]
        self.current_location = self.locations[np.random.randint(self.locations_num)]
 
       
  
    def location_function(self):
        if self.current_location == "You are in the living_room": 
           if self.agent_action == 'go east':
              self.current_location = "You are in the garden"
           if self.agent_action == 'go south':
              self.current_location = "You are in the bedroom"

        if self.current_location == "You are in the garden":
           if self.agent_action == 'go east':
              self.current_location = "You are in the kitchen"
           if self.agent_action == 'go west':
              self.current_location = "You are in the living_room"
           if self.agent_action == 'go south': 
              self.current_location = "You are in the bathroom"

        if self.current_location == "You are in the kitchen":
           if self.agent_action == 'go west':
              self.current_location = "You are in the garden" 

        if self.current_location == "You are in the bedroom":
           if self.agent_action == 'go north':
              self.current_location = "You are in the garden"
           if self.agent_action == 'go east':
              self.current_location = "You are in the bathroom"

        if self.current_location == "You are in the bathroom":
           if self.agent_action == 'go north':
              self.current_location = "You are in the garden"
           if self.agent_action == 'go west':
              self.current_location = "You are in the bedroom"

    def reward_function(self):
        if self.current_quest == "You are hungry": # hungry, kitchen
             if self.current_location ==  "You are in the kitchen" and self.agent_action == 'eat apple': 
                self.reward = self.step_reward
             else:
                self.reward = self.step_negative_reward

        if self.current_quest == "You are sleepy": # bedroom, tired
             if self.current_location ==  "You are in the bedroom" and self.agent_action == 'sleep bed': 
                self.reward = self.step_reward
             else:
                self.reward = self.step_negative_reward
 
        if self.current_quest == "You are bored": # living, bored
             if self.current_location ==  "You are in the living_room" and self.agent_action == 'watch TV': 
                self.reward = self.step_reward
             else:
                self.reward = self.step_negative_reward

        if self.current_quest == 'You are getting fat':  # garden, fat
             if self.current_location == "You are in the garden" and self.agent_action == 'exercise body': 
                self.reward = self.step_reward
             else:
                self.reward = self.step_negative_reward
       
        if self.current_quest == "You are dirty":  # toilet, dirty
             if self.current_location == "You are in the bathroom" and self.agent_action == 'wash' and self.objects == 'body':
                self.reward = self.step_reward
             else:
                self.reward = self.step_negative_reward

       

    def get_state_reward(self,act): 
               
        self.agent_action = act

        self.location_function() 
        self.reward_function()
       
        return self.current_quest+';'+self.current_location,self.reward
        
