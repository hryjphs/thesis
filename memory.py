# -*- coding: utf-8 -*-
"""
Replay buffer
Prioritzed Replay buffer -- to be continued
"""
import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, input_dims):    #max memory size: max_size
        self.mem_size = max_size
        self.mem_cntr = 0   #keep track of our first unsaved memory 

        self.state_memory = np.zeros((self.mem_size, input_dims),   # * to unpack the list to elements
                                    dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_dims),
                                dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size  
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = 1 - int(done)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)  #the random sample is generated as if a were np.arange(max_mem), shape = batch_size, sample without replace

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal