import numpy as np
import pandas as pd
import os
import time
import string
import random
import sys
import SPEC

# description : quest. location. previous action. mislead (not including quest)
# 4 sentences / 20 words

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

class HomeWorld():
    def __init__(self, num_home_rooms=5, seq_length=20, max_step=5):
        self.regions = ["home", "school"]
        self.home_rooms = ["livingroom", "garden", "kitchen", "bedroom", "toilet"]
        self.school_rooms = ["physics_classroom", "math_classroom", "music_classroom", "canteen", "field", "library"]
        self.home_actions = ["eat", "sleep", "watch", "exercise","wash","go"]
        self.school_actions =  ["study", "bath", "borrow", "attend"]
        self.objects = ["north", "south", "east", "west", "apple", "bed", "body","face","tv"]
        self.home_quests = ["you are hungry", "you are sleepy", "you are bored", "you are getting fat", "you are dirty"]
        self.school_quests = ["You are going to school", "You are going to home",
                       "You are going to borrow books",
                       "You are going to attend math class", "You are going to attend physics class",
                       "You are going to attend music class"]
        self.home_quests_mislead = ["you are not hungry", "you are not sleepy", "you are not bored","you are not getting fat",
                               "you are not dirty"]
        self.school_quests_mislead = ["You are not going to school", "You are not going to home",
                               "You are not going to borrow books",
                               "You are not going to attend music class"]
        self.idx2word = ["not", "but", "now"]

    def get_random_location(self):
        return random.choice(self.home_rooms)
    def get_random_quest(self):
        return random.choice(self.home_quests)
    def get_home_quests_mislead(self):
        return random.choice(self.home_quests_mislead)

    def read_file(self):
        fp = open("description.txt")
        lines = fp.readlines()
        for room in self.home_rooms:
            if room == "livingroom":
                temp = lines[1] + lines[2]
            elif room == "garden":
                temp = lines[5] + lines[6]
            elif room == "kitchen":
                temp = lines[9] + lines[10] + lines[11]
            elif room == "bedroom":
                temp = lines[14]
            else:
                temp = lines[17]
            description[room] = temp

    def new_game(self):
        self.first_quest = self.get_random_quest()
        self.read_file()
        self.first_location = self.get_random_location()

        print ('The game is now started')
        print ('Welcome to our Home World!!')
        print (description[self.first_location])
        print ('here is your quest: '+self.first_quest)

        return self.first_quest,self.first_location


def display (action_list):
    global agent_location
    global agent_quest
    global agent_mislead

    agent_mislead = RL_environment.get_home_quests_mislead()
    print ('your current quest is ' + agent_quest + '.' + ' you are currently in ' + agent_location + '.' + ' you just did ' + action_list[0] + " " + action_list[1] + '. ' + agent_mislead + '.' )

def check_location(action_list):
    global agent_location
    global agent_quest
    global total_reward
    global quest_checklist

    room = agent_location
    #print ('action_list:', action_list)
    #print ('room:', room)
    if room == 'livingroom':
        #print ('1')
        if action_list[0] == 'go' and action_list[1] == 'east':
            agent_location = 'garden'
            total_reward = total_reward + reward_step_quest_not_complete
            quest_checklist = False
        elif action_list[0] == 'go' and action_list[1] == 'south':
            agent_location = 'bedroom'
            total_reward = total_reward + reward_step_quest_not_complete
            quest_checklist  = False
        elif agent_quest == RL_environment.home_quests[2] and action_list[0] == 'watch' and action_list[1] == 'tv':
            total_reward = total_reward + reward_quest_complete
            quest_checklist = True
        else:
            total_reward = total_reward + reward_step_quest_not_complete
            agent_location = room
            quest_checklist = False
    if room == 'garden':
        #print ('2')
        if action_list[0] == 'go' and action_list[1] == 'west':
            agent_location = 'livingroom'
            total_reward = total_reward + reward_step_quest_not_complete
        elif action_list[0] == 'go' and action_list[1] == 'east':
            agent_location = 'kitchen'
            total_reward = total_reward + reward_step_quest_not_complete
        elif action_list[0] == 'go' and action_list[1] == 'south':
            agent_location = 'toilet'
            total_reward = total_reward + reward_step_quest_not_complete
        elif agent_quest == RL_environment.home_quests[3] and action_list[0] == 'exercise' and action_list[1] == 'body':
            total_reward = total_reward + reward_quest_complete
            quest_checklist = True
        else:
            total_reward = total_reward + reward_step_quest_not_complete
            agent_location = room
            quest_checklist = False
    if room == 'kitchen':
        #print ('3')
        if action_list[0] == 'go' and action_list[1] == 'west':
            agent_location = 'garden'
            total_reward = total_reward + reward_step_quest_not_complete
        elif agent_quest == RL_environment.home_quests[0] and action_list[0] == 'eat' and action_list[1] == 'apple':
            total_reward = total_reward + reward_quest_complete
            quest_checklist = True
        else:
            total_reward = total_reward + reward_step_quest_not_complete
            agent_location = room
            quest_checklist = False
    if room == 'bedroom':
        #print ('4')
        if action_list[0] == 'go' and action_list[1] == 'east':
            agent_location = 'toilet'
            total_reward = total_reward + reward_step_quest_not_complete
        elif action_list[0] == 'go' and action_list[1] == 'north':
            agent_location = 'livingroom'
            total_reward =  total_reward + reward_step_quest_not_complete
        elif agent_quest == RL_environment.home_quests[1] and action_list[0] == 'sleep' and action_list[1] == 'bed':
            total_reward = total_reward + reward_quest_complete
            quest_checklist = True
        else:
            total_reward = total_reward + reward_step_quest_not_complete
            agent_location = room
            quest_checklist = False

    if room == 'toilet':
        #print ('5')
        if action_list[0] == 'go' and action_list[1] == 'west':
            agent_location = 'bedroom'
            total_reward = total_reward + reward_step_quest_not_complete
        elif action_list[0] == 'go' and action_list[1] == 'north':
            agent_location = 'garden'
            total_reward = total_reward + reward_step_quest_not_complete
        elif agent_quest == RL_environment.home_quests[4] and action_list[0] == 'wash' and action_list[1] == 'face':
            total_reward = total_reward + reward_quest_complete
            quest_checklist = True
        else:
            total_reward = total_reward + reward_step_quest_not_complete
            agent_location = room
            quest_checklist = False
    display(action_list)

def handle_action(action):
    action_list = action.split()
    if len(action_list) == 2:
        check_location(action_list)

    else:
        print ('try to input two words action command')
"""
time_step = 100
reward_step_quest_not_complete = -0.01
reward_quest_complete = 3
total_reward = 0

RL_environment = HomeWorld()

for i in range(time_step):
    if i == 0 and quest_checklist == False:
        agent_quest, agent_location = RL_environment.new_game()

    if quest_checklist == True and i != 0:
        quest_completed_count = quest_completed_count + 1
        agent_quest = RL_environment.get_random_quest()
        agent_quest, agent_location = RL_environment.new_game()

    input_command = raw_input('What do you want to do? : ')
    handle_action(input_command)
"""
