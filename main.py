import Environment

time_step = 5
RL_environment = Environment.HomeWorld()

print ('The game is now started')
RL_environment.new_game()
for i in range(time_step):
    if i == 0:
        agent_location = RL_environment.first_location
    else:
        pass
    input_command = input('What do you want to do? : ')
    print ('agent_location inside for loop: ', agent_location)
    handle_action(input_command)

    # print input_command
