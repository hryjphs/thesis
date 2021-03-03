# -*- coding: utf-8 -*-
"""
A trading environment, which contains
(1) observation space: already simulated samples / or true market data,
(2) action space: continuous or discrete, holdings of the hedging assets
(3) reset(): go to a new episode
(4) step(): transist to the next state

"""

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from simulation import get_sim_path, get_sim_path_sabr


class TradingEnv(gym.Env):
    """
    trading environment;
    contains observation space (already simulated samples), action space, reset(), step()
    """

    def __init__(self, cash_flow_flag=0, dg_random_seed=1, num_sim=500002, sabr_flag = False,
        continuous_action_flag=False, ticksize=0.1, multi=1, init_ttm=5, trade_freq=1, num_contract=1, sim_flag = True,
        mu=0.05, vol=0.2, S=100, K=100, r=0, q=0, beta=1, rho=-0.4, volvol = 0.6, ds = 0.001, k=0):
        """ cash_flow_flag: 1 if reward is defined using cash flow, 0 if profit and loss; dg_random_seed: random seed for simulation;
            num_sim: number of paths to simulate; sabr_flag: whether use sabr model; continuous_action_flag: continuous or discrete
            action space; init_ttm: initial time to maturity in unit of day; trade_freq: trading frequency in unit of day;
            num_contract: number of call option to short; sim_flag = simulation or market data;
            Assume the trading cost include spread cost and price impact cost, cost(n)= multi*ticksize*(|n|+0.01*n^2)
            Reward is about ((change in value) - k/2 * (change in value)^2 ), where k is risk attitude, k=0 if risk neutral
        """
        # observation
        if sim_flag:
            # generate data now
            if sabr_flag:
                self.path, self.option_price_path, self.delta_path, self.bartlett_delta_path = get_sim_path_sabr(M=init_ttm, freq=trade_freq, 
                                                                                                                 np_seed=dg_random_seed, num_sim=num_sim, mu=0.05, vol=0.2, S=100, K=100, r=0, q=0,
                                                                                                                  beta=1, rho=-0.4, volvol = 0.6, ds = 0.001)
            
            else:
                self.path, self.option_price_path, self.delta_path = get_sim_path(M=init_ttm, freq=trade_freq,
                                                                                  np_seed=dg_random_seed, num_sim=num_sim, mu=0.05, vol=0.2, S=100, K=100, r=0, q=0)
        else:
            # use actual data ---> to be continued
            return 
        
        # other attributes
        self.num_path = self.path.shape[0]

        # set num_period: initial time to maturity * daily trading freq + 1 (see get_sim_path() in simulation.py): (s0,s1...sT) -->T+1
        self.num_period = self.path.shape[1]
        # print("***", self.num_period)

        # time to maturity array
        self.ttm_array = np.arange(init_ttm, -trade_freq, -trade_freq)
        # print(self.ttm_array)

        # cost part
        self.ticksize = ticksize  
        self.multi = multi

        # risk attitude
        self.k = k                                                                                 

        # step function initialization depending on cash_flow_flag
        if cash_flow_flag == 1:
            self.step = self.step_cash_flow   # see step_cash_flow() definition below. Internal reference use self.
        else:
            self.step = self.step_profit_loss

        self.num_contract = num_contract
        self.strike_price = 100

        # track the index of simulated path in use
        self.sim_episode = -1

        # track time step within an episode (it's step)
        self.t = None

        # action space for holding 
        # With L contracts, each for 100 shares, one would not want to trade more than 100·L shares                                                                                                       #action space justify?
        if continuous_action_flag:
            self.action_space = spaces.Box(low=np.array([0]), high=np.array([num_contract * 100]), dtype=np.float32)
        else:
            self.num_action = num_contract * 100 + 1
            self.action_space = spaces.Discrete(self.num_action)  #number from 0 to self.num_action-1

        # state element, assumed to be 3: current price, current holding, ttm
        self.num_state = 3

        self.state = []    # initialize current state

        # seed and start
        self.seed()  # call this function when intialize ...
        # self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)  #self.np_random now is a generateor np.random.RandomState() with a strong random seed; seed is a strong random seed
        return [seed]

    def reset(self):
        # repeatedly go through available simulated paths (if needed)
        # start a new episode
        self.sim_episode = (self.sim_episode + 1) % self.num_path

        self.t = 0

        price = self.path[self.sim_episode, self.t]
        position = 0

        ttm = self.ttm_array[self.t]

        self.state = [price, position, ttm]

        return self.state

    def step_cash_flow(self, action):
        """
        cash flow period reward
        take a step and return self.state, reward, done, info
        """

        # do it consistently as in the profit & loss case
        # current prices (at t)
        current_price = self.state[0]

        # current position
        current_position = self.state[1]

        # update time/period
        self.t = self.t + 1

        # get state for tomorrow
        price = self.path[self.sim_episode, self.t]
        position = action
        ttm = self.ttm_array[self.t]

        self.state = [price, position, ttm]   #state transist to next price, ttm and stores current action(position)

        # calculate period reward (part 1)
        cash_flow = -(position - current_position) * current_price - (np.abs(position - current_position) +0.01*(position - current_position)**2) * self.ticksize * self.multi    

        # if tomorrow is end of episode, only when at the end day done=True , self.num_period = T/frequency +1
        if self.t == self.num_period - 1:
            done = True   #you have arrived at the terminal
            # add (stock payoff + option payoff) to cash flow
            cash_flow = cash_flow + price * position - max(price - self.strike_price, 0) * self.num_contract * 100 - (position + 0.01*position**2) * self.ticksize * self.multi  
            reward = cash_flow -self.k /2 * (cash_flow)**2
        else:
            done = False
            reward = cash_flow -self.k /2 * (cash_flow)**2

        # for other info
        info = {"path_row": self.sim_episode}

        return self.state, reward, done, info

    def step_profit_loss(self, action):
        """
        profit loss period reward
        """

        # current prices (at t)
        current_price = self.state[0]
        current_option_price = self.option_price_path[self.sim_episode, self.t]

        # current position
        current_position = self.state[1]

        # update time
        self.t = self.t + 1

        # get state for tomorrow (at t + 1)
        price = self.path[self.sim_episode, self.t]
        option_price = self.option_price_path[self.sim_episode, self.t]
        position = action
        ttm = self.ttm_array[self.t]

        self.state = [price, position, ttm]

        # calculate period reward (part 1)
        reward = (price - current_price) * position - (np.abs(position - current_position) +0.01*(position - current_position)**2) * self.ticksize * self.multi 

        # if tomorrow is end of episode
        if self.t == self.num_period - 1:
            done = True
            reward = reward - (max(price - self.strike_price, 0) - current_option_price) * self.num_contract * 100 - (position + 0.01*position**2) * self.ticksize * self.multi   #liquidate option and stocks
            reward = reward - self.k / 2 * (reward)**2
        else:
            done = False
            reward = reward - (option_price - current_option_price) * self.num_contract * 100
            reward = reward - self.k / 2 * (reward)**2
        # for other info later
        info = {"path_row": self.sim_episode}

        return self.state, reward, done, info