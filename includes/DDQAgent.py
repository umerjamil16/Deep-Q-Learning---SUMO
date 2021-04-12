import traci
import random
import statistics as stat

import numpy as np
import torch as T
from includes.deep_q_network import DeepQNetwork
from includes.replay_memory import ReplayBuffer

class DDQAgent(object):
    """
        This is single agent class.
    """
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, algo=None, env_name=None, chkpt_dir='tmp/dqn', TLC_name="gneJ26"):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        self.q_eval = DeepQNetwork(self.lr, self.n_actions,
                                    input_dims=self.input_dims,
                                    name=self.env_name+'_'+self.algo+'_q_eval',
                                    chkpt_dir=self.chkpt_dir)

        self.q_next = DeepQNetwork(self.lr, self.n_actions,
                                    input_dims=self.input_dims,
                                    name=self.env_name+'_'+self.algo+'_q_next',
                                    chkpt_dir=self.chkpt_dir)

        self.state = []


        ## Defining cardinal parameters
        # Accumulated traffic queues in each direction
        self.no_veh_N = 0
        self.no_veh_E = 0
        self.no_veh_W = 0
        self.no_veh_S = 0

        # Accumulated wait time in each direction
        self.wait_time_N = 0
        self.wait_time_E = 0
        self.wait_time_W = 0
        self.wait_time_S = 0
        
        self.TLC_name = TLC_name
        self.threshold = 0

        self.acc_wait_time = []

        self.reward = 0

        self.switch_time = 0
        self.next_switch = 0


    def get_reward(self):

        current_wait_time = self.get_avg_wait_time()
        if len(self.acc_wait_time) == 0:
            self.reward = 0
        else:
            #reward_clip = self.acc_wait_time[-1] - current_wait_time
            self.reward = self.acc_wait_time[-1] - current_wait_time
            #np.clip(np.array([reward_cl]), -1, 1)[0]
        
        self.acc_wait_time.append(current_wait_time)

        if len(self.acc_wait_time) > 3: # to reduce the array size  / keep only last three elements
            self.acc_wait_time = self.acc_wait_time[-3:]

        return self.reward

    def get_avg_wait_time(self):
        phase_central = traci.trafficlight.getRedYellowGreenState(self.TLC_name)
            ####
        for vehicle_id in traci.vehicle.getIDList():
            vehicle_speed = traci.vehicle.getSpeed(vehicle_id)
            road_id = traci.vehicle.getRoadID(vehicle_id)
            vehicle_wait_time = traci.vehicle.getAccumulatedWaitingTime(vehicle_id)
            #print("Vehicle ID: ", vehicle_id, "Speed: ", vehicle_speed, "Road Id: ", road_id)

            #Count vehicle at the TLC junction
            if vehicle_speed < 1: # Count only stoped vehicles
                if   road_id == "N02TL0": #North
                    self.no_veh_N += 1 
                    self.wait_time_N += vehicle_wait_time                    
                elif road_id == "E02TL0": #East
                    self.no_veh_W += 1 
                    self.wait_time_W += vehicle_wait_time
                elif road_id == "W02TL0": #West
                    self.no_veh_E += 1 
                    self.wait_time_E += vehicle_wait_time
                elif road_id == "S02TL0": #South
                    self.no_veh_S += 1 
                    self.wait_time_S += vehicle_wait_time

        return stat.mean([self.wait_time_N, self.wait_time_E, self.wait_time_W, self.wait_time_S])

    def step(self, action, step, env):
        print("In step func")
        print("Action: ", action)

        self.reset_lane_traffic_info_params()

        # 1st APPLY the choosed action
        if action == 0:
            traci.trafficlight.setRedYellowGreenState(self.TLC_name, "GrrrrGrrrrGGGGGGrrrr")
            #traci.trafficlight.setPhaseDuration(self.TLC_name, 15)
            print("Taking action 0")
#east
        elif action == 1:
            traci.trafficlight.setRedYellowGreenState(self.TLC_name, "GrrrrGrrrrGrrrrGGGGG")
            #traci.trafficlight.setPhaseDuration(self.TLC_name, 15)
            print("Taking action 1")
#west
        elif action == 2:
            traci.trafficlight.setRedYellowGreenState(self.TLC_name, "GrrrrGGGGGGrrrrGrrrr")
            #traci.trafficlight.setPhaseDuration(self.TLC_name, 15)
            print("Taking action 2")
#south
        elif action == 3:
            traci.trafficlight.setRedYellowGreenState(self.TLC_name, "GGGGGGrrrrGrrrrGrrrr")
            #traci.trafficlight.setPhaseDuration(self.TLC_name, 15)
            print("Taking action 3")


        env.simulationStep()
            
        # 2nd, find the Reward 
        #func will be implemented later
        reward = self.get_reward()
        
        # 3rd: find new state
        # return state, reward, info_dict
        state = self.get_state()
#        print("State vec in step(): ", state)
 #       print("length of State vec in step(): ", np.shape(state))
        
        #state = [self.wait_time_N, self.wait_time_E, self.wait_time_W, self.wait_time_S,
         #        self.no_veh_N, self.no_veh_E, self.no_veh_W, self.no_veh_S]
        
        # 4th, an info dict
        info = {"TLC_name": self.TLC_name}

        return state, reward, info


        #zero all the class variable

    def get_state_vector(self, num_of_cars):
        #print("here")
        car_lim = 300
        if car_lim == 200:
            BW = [0, 1,1,1, 2,2,2, 4,4, 8,8,8, 10]
            AC = [0, 4,8,12, 20,28,36, 52,68, 100,132,164, 204]
        elif car_lim == 300:
            BW = [0, 1,1, 2,2, 4,4, 8,8, 10, 12,12,12]
            AC = [0, 4,8, 16,24, 40,56, 88,120, 160, 208,256,304]
        elif car_lim == 500:
            BW = [0, 1, 2, 8, 12, 16, 24, 28, 32] # Weights Block
            AC = [0, 4, 20, 52, 100, 164, 260, 372, 500] # Accumated Cars
        else:
            print('Undefine car number state')

        state = np.zeros((4,len(AC)-1), dtype=np.uint8)
        if num_of_cars == 0:
            #print(state)
            return state.flatten()
        else:
            for i in range(1, len(BW)):
                if num_of_cars <= AC[i]:
                    fils = int(np.round(((num_of_cars - AC[i-1]) / BW[i]) + 0.1))
                    state[:fils, i-1] = 1
                    return state.flatten()
                else:
                    state[:,i-1] = 1
            return state.flatten()

    def get_state(self):
        self.reset_lane_traffic_info_params()
        
        phase_central = traci.trafficlight.getRedYellowGreenState(self.TLC_name)
        ####
        for vehicle_id in traci.vehicle.getIDList():
            vehicle_speed = traci.vehicle.getSpeed(vehicle_id)
            road_id = traci.vehicle.getRoadID(vehicle_id)
            vehicle_wait_time = traci.vehicle.getAccumulatedWaitingTime(vehicle_id)
            #print("Vehicle ID: ", vehicle_id, "Speed: ", vehicle_speed, "Road Id: ", road_id)

            #Count vehicle at the TLC junction
            #Count vehicle at the TLC junction
            if vehicle_speed == 0: # Count only stoped vehicles
                if   road_id == "N02TL0": #North
                    self.no_veh_N += 1 
                    self.wait_time_N += vehicle_wait_time                    
                elif road_id == "E02TL0": #East
                    self.no_veh_W += 1 
                    self.wait_time_W += vehicle_wait_time
                elif road_id == "W02TL0": #West
                    self.no_veh_E += 1 
                    self.wait_time_E += vehicle_wait_time
                elif road_id == "S02TL0": #South
                    self.no_veh_S += 1 
                    self.wait_time_S += vehicle_wait_time

            state = self.get_state_vector(self.no_veh_E)
            state = np.append(state, self.get_state_vector(self.no_veh_W))
            state = np.append(state, self.get_state_vector(self.no_veh_N))
            state = np.append(state, self.get_state_vector(self.no_veh_S))
            self.state = state

            #print("Size of state space: ", np.shape(state))


        state_space = [self.wait_time_N, self.wait_time_W, self.wait_time_E, self.wait_time_S,
                 self.no_veh_N, self.no_veh_W, self.no_veh_E, self.no_veh_S]
        
        #return self.state#
        return np.array(state_space)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            print("Taking proper action")
            state = T.tensor(observation,dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            print("Taking random action")
            action = np.random.choice(self.action_space)


        return action

    def reset_lane_traffic_info_params(self):
        # Accumulated traffic queues in each direction
        self.no_veh_N = 0
        self.no_veh_E = 0
        self.no_veh_W = 0
        self.no_veh_S = 0

        # Accumulated wait time in each direction
        self.wait_time_N = 0
        self.wait_time_E = 0
        self.wait_time_W = 0
        self.wait_time_S = 0

    def decrement_epsilon(self):

        self.epsilon = self.epsilon - self.eps_dec \
                        if self.epsilon > self.eps_min else self.eps_min
                        

    def store_transition(self, state, action, reward, state_):
        self.memory.store_transition(state, action, reward, state_)

    def sample_memory(self):
        state, action, reward, new_state = \
                                self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        return states, actions, rewards, states_

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                           if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_ = self.sample_memory()

        indices = np.arange(self.batch_size)

        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_)
        q_eval = self.q_eval.forward(states_)

        max_actions = T.argmax(q_eval, dim=1)
        #q_next[dones] = 0.0

        q_target = rewards + self.gamma*q_next[indices, max_actions]
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()

        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

    """

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_ = self.sample_memory()
        indices = np.arange(self.batch_size)

        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_).max(dim=1)[0]

        #q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        #self.decrement_epsilon()
    """
        
    def printStatusReport(self, step):
        # Print status report
        phase_central = traci.trafficlight.getRedYellowGreenState(self.TLC_name)

        print("--- Status Report ---")
        print("Step: ", step)
        print("Signal State: ", phase_central)
        print("Last switch time at action: ", self.switch_time)
        print("Get next switch: ", (-self.switch_time + traci.trafficlight.getNextSwitch(self.TLC_name)))
        print("Get phase duration: ", (-self.switch_time + traci.trafficlight.getPhaseDuration(self.TLC_name)))


        print("no_veh_N: ", self.no_veh_N)
        print("no_veh_E: ", self.no_veh_E)
        print("no_veh_W: ", self.no_veh_W)
        print("no_veh_S: ", self.no_veh_S)

        print("wait_time_N", self.wait_time_N)
        print("wait_time_E", self.wait_time_E)
        print("wait_time_W", self.wait_time_W)
        print("wait_time_S", self.wait_time_S)

