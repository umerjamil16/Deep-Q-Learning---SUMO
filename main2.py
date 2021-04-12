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
    env = SimEnv()


    for episode in range(1, 100):
        #Need to reset traffic as well
        print("Episode: ", episode)
        traffic = GenerateTraffic(6000)
        traffic.set_traffic_flow(100, 1)
        
        #restart simulation as
        n_steps = 0
        #env.close_sumo() 
        env.start_sumo()

        for i in range(100):
            time.sleep(0)

            env.simulationStep()

            n_steps+=1

        env.close_sumo() 

        
