from includes.sumo import SimEnv

import time
import traci
import numpy as np
import matplotlib.pyplot as plt
import statistics as stat


from includes.DDQAgent import DDQAgent
from includes.utils import plot_learning_curve

from traffic_gen import GenerateTraffic
import random

def save_plot_neg(list1, episode):
    print("Saving plot")
    x = [i for i in range(len(list1))]

    plt.plot(x, list1)
    plt.grid()
    plt.savefig("plots21_march/neg_reward_episodes_"+str(episode)+".png") # save as png
    plt.close()

def save_plot_pos(list1, episode):
    print("Saving plot")
    x = [i for i in range(len(list1))]

    plt.plot(x, list1)
    plt.grid()
    plt.savefig("plots21_march/pos_reward_episodes_"+str(episode)+".png") # save as png
    plt.close()

def save_plot_reward(list1, episode):
    print("Saving plot")
    x = [i for i in range(len(list1))]

    plt.plot(x, list1)
    plt.grid()
    plt.savefig("plots21_march/reward_episodes_"+str(episode)+".png") # save as png
    plt.close()

def save_plot(list1, episode):
    print("Saving plot")
    x = [i for i in range(len(list1))]

    plt.plot(x, list1)
    plt.grid()
    plt.savefig("plots21_march/"+str(episode)+".png") # save as png
    plt.close()

def single_step(agent, n_steps, episode, observation, env):
    score= 0
    action = agent.choose_action(observation)
    observation_, reward, info = agent.step(action, n_steps, env)
    score += reward
    """
    if not load_checkpoint:# and n_steps>100:
        agent.store_transition(observation, action, reward, observation_)
        agent.learn()
    """         
    observation = observation_

    #if n_steps > 100:
    scores.append(score)
    steps_array.append(n_steps)
        
    avg_score = np.mean(scores[-100:])
    #print('-- episode: ', episode, 'steps', n_steps)

    return observation, reward, action


if __name__ == '__main__':
    env = SimEnv()
    #env.start_sumo()
        
    delayTime = 0#1

    Central = "TL0"
    n_games = 5200
    n_steps = 0

    scores = []
    eps_history = []
    steps_array = []

    best_score = -np.inf
    load_checkpoint = False

    episodes = 1500

    EPS = (1-0.05)/episodes

    agent = DDQAgent(gamma=0.99, epsilon=1, lr=0.0001,
                     input_dims=8,
                     n_actions=4, mem_size=50000, eps_min=0.01,
                     batch_size=64, replace=1000, eps_dec=EPS,
                     chkpt_dir='models/', algo='DQNAgent',
                     env_name='SUMO_tlc', TLC_name = Central)

    if load_checkpoint:
        agent.load_models()

    episodic_reward_neg =[]
    episodic_reward_pos =[]
    episodic_reward_sum =[] 
    episodic_reward_avg =[] 

    #env.start_sumo()
    
    eps_history = []
    
    for episode in range(1, episodes):
        neg_reward_current_episode = []
        pos_reward_current_episode = []
        reward_current_episode = []
        n_steps = 0

        # Reset Traffic
        traffic = GenerateTraffic(n_games+600)
        traffic_flow = 0#random.randint(0,3)
        print("Generating traffic for episode: ", episode, " and flow: ", traffic_flow)
        traffic.set_traffic_flow(episode, traffic_flow)

        #Start Sumo
        
        env.start_sumo()

        obs, reward, info = agent.step(0, n_steps, env) #taking random action
        action = 0
        reward_ = agent.get_reward()
        start_time = time.time()

        green_time = 30
        
        busy_count = green_time + n_steps

        for i in range(n_games):
            if n_steps > busy_count:

                observation_ = agent.get_state()
                reward_ = agent.get_reward()
                print("\n -------------------------\n")

                print('-- episode: ', episode, 'steps', n_steps, "Reward: ", reward_)

                if not load_checkpoint:# and n_steps>100:
                    agent.store_transition(obs, action, reward_, observation_)
                    agent.learn()

#                print("Performing action")
                obs, reward, action = single_step(agent, n_steps, episode, observation_, env)
                #observation = obs

                busy_count = green_time + n_steps
                #print("Busy count: ", busy_count)

#                print("obs_ac: ", obs)

                if reward_ < 0:# and n_steps > 100:
                    neg_reward_current_episode.append(reward_)
                if reward_ >0:# and n_steps > 100:
                    pos_reward_current_episode.append(reward_)

                if True:#n_steps > 100: 
                    reward_current_episode.append(reward_)

                time.sleep(delayTime)
                #env.simulationStep()
                n_steps += 1
            else:
                #print("Agent is busy: ", n_steps-busy_count)
                n_steps += 1
                env.simulationStep()
#                print("Obs: ", observation_)

        # Plots for each episode
        #save_plot(neg_reward_current_episode, str(episode)+"_neg_subreward")
        #save_plot(pos_reward_current_episode, str(episode)+"_pos_subreward")
        #save_plot(reward_current_episode, str(episode)+"_overall_subreward")


        print("######################################################")
        print('episode: ', episode,'neg_reward: ', sum(neg_reward_current_episode),
        'pos_reward: ', sum(pos_reward_current_episode),
        'cumm reward: ', sum(reward_current_episode)
        )
        print("Time taken (minutes): ", ((time.time() - start_time)/60))
        print("Generating new traffic scenario")
        print("######################################################")

        # Plots for overall episodes
        episodic_reward_neg.append(sum(neg_reward_current_episode))
        episodic_reward_pos.append(sum(pos_reward_current_episode))
        episodic_reward_sum.append(sum(reward_current_episode))
        episodic_reward_avg.append(stat.mean(reward_current_episode))
        
        #env.close_sumo() 
        #if episode %10 == 0:
        save_plot(episodic_reward_neg, str(episode)+"_neg")
        #save_plot(episodic_reward_pos, str(episode)+"_pos")
        #save_plot(episodic_reward_sum, str(episode)+"_sum")
        #save_plot(episodic_reward_avg, str(episode)+"_avg")
        
        agent.save_models()

        eps_history.append(agent.epsilon)
        save_plot_neg(eps_history, str(episode)+"_eps")

        agent.decrement_epsilon()

        env.close_sumo() 
    


    #print("Episodic reward list: ", episodic_reward_neg)
#    save_plot(episodic_reward_neg, "Final")
    save_plot_neg(episodic_reward_neg, "Final")
    #save_plot_pos(episodic_reward_pos, "Final")
    #save_plot_reward(episodic_reward_sum, "Final")