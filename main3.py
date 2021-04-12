from includes.sumo import SimEnv

import time
import traci
import numpy as np
import matplotlib.pyplot as plt

from includes.DDQAgent import DDQAgent
from includes.utils import plot_learning_curve


from traffic_gen import GenerateTraffic

def save_plot(list1, episode):
    print("Saving plot")
    x = [i for i in range(len(list1))]

    plt.plot(x, list1)
    plt.savefig("neg_reward_episodes_"+str(episode)+".png") # save as png



if __name__ == '__main__':
    step = 0
    delayTime = 0#1/8

    Central = "TL0"
    n_games = 300#5200
    episodes = 11##00#300

    episodic_reward = []
    neg_reward_current_episode = []

    n_steps = 0

    scores = []
    eps_history = []
    steps_array = []

    best_score = -np.inf
    load_checkpoint = False

    agent = DDQAgent(gamma=0.99, epsilon=1, lr=0.0001,
                     input_dims=192,
                     n_actions=4, mem_size=50000, eps_min=0.1,
                     batch_size=32, replace=200, eps_dec=1e-5,
                     chkpt_dir='models/', algo='DQNAgent',
                     env_name='SUMO_tlc', TLC_name = Central)

    if load_checkpoint:
        agent.load_models()



    for episode in range(1, episodes):
        print("Episode: ", episode)
        #Need to reset traffic as well
        traffic = GenerateTraffic(6000)
        traffic.set_traffic_flow(100, 1)
        
        #restart simulation as
        n_steps = 0
        #env.close_sumo() 
        env = SimEnv()        
        env.start_sumo()
            
        observation, reward, info = agent.step(0, n_steps) #taking random action
        neg_reward_current_episode = []

        #print("Obs: ", observation)
        #print("len Obs: ", len(observation))

        for i in range(n_games):

            action = agent.choose_action(observation)
            observation_, reward, info = agent.step(action, n_steps)

            if reward < 0:
                neg_reward_current_episode.append(reward)
        
            if n_steps > 30:
                print("Learning")
                if not load_checkpoint:
                    agent.store_transition(observation, action, reward, observation_)
                    agent.learn()
                    
            observation = observation_
            n_steps += 1
            print("Episode: ", episode, 'Game: ', i,'neg_reward: ', reward)


            time.sleep(delayTime)
            env.simulationStep()

        env.close_sumo() 

        print("###########################")
        print('episode: ', episode,'neg_reward: ', sum(neg_reward_current_episode))
        episodic_reward.append(sum(neg_reward_current_episode))
        print("###########################")

        episodic_reward.append(sum(neg_reward_current_episode))

        if episode %10 == 0:
            save_plot(episodic_reward, episode)


    #print("Episodic reward list: ", episodic_reward)
    save_plot(episodic_reward, "final")