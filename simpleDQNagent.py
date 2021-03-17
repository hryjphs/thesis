# -*- coding: utf-8 -*-
"""
This is a baseline model using DQN + Replaybuffer + discrete action 
More models should be considered later....
"""

from memory import ReplayBuffer 
import numpy as np
import keras
import tensorflow as tf

def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):    # all connected; otherwise define your own class
    model = keras.Sequential([
        keras.layers.Dense(fc1_dims, activation='relu'),
        keras.layers.Dense(fc2_dims, activation='relu'),
        keras.layers.Dense(n_actions, activation=None)])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mean_squared_error')

    return model

class Agent():
    def __init__(self, lr, gamma, epsilon, batch_size,
                input_dims, env, epsilon_dec=1e-3, epsilon_end=0.01,
                mem_size=1000, fname='dqn_model.h1'):
        self.env = env
        self.action_space = self.env.action_space     #discrete
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = epsilon_dec
        self.eps_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.n_actions = self.env.num_action 
        self.q_eval = build_dqn(lr, self.n_actions, input_dims, 256, 256)
        

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, observation):  
        # epsilon greedy to choose action, maybe later can try out Boltzman... 
        if np.random.random() < self.epsilon:
            action = self.action_space.sample() #random select an action from action space
        else:
            state = np.array([observation])
            actions = self.q_eval.predict(state)  # Q(a,s)
            action = np.argmax(actions)

        return action

    def learn(self):
        # DQN 
        if self.memory.mem_cntr < self.batch_size:
            return   

        states, actions, rewards, states_, dones = \
                self.memory.sample_buffer(self.batch_size)

        

        #actions = actions.reshape(-1, 1)
        #rewards = rewards.reshape(-1, 1)
        #dones = dones.reshape(-1, 1)
        

        q_eval = self.q_eval.predict(states)  # Q(a,s) for all actions, for all states in batch
        
        q_next = self.q_eval.predict(states_)  # Q(a,s_) for all actions....


        q_target = np.copy(q_eval)
        batch_index = np.arange(self.batch_size, dtype=np.int32)  # 0-63
        

        q_target[batch_index, actions] = rewards + \
                        self.gamma * np.max(q_next,1)*dones
        


        self.q_eval.train_on_batch(states, q_target)

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > \
                self.eps_min else self.eps_min

    def save_model(self):
        self.q_eval.save(self.model_file)


    def load_model(self):
        self.q_eval = load_model(self.model_file)


