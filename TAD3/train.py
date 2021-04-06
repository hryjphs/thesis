# -*- coding: utf-8 -*-
"""
Initial time to maturity is 10 days; Assume we could trade 5 times a day; short 1 European call option contract ( each for 100 shares)
Geometric brownian simulation (constant volatility) or SABR (stochastic volatility) to simulate the underlying stock prices
Action space: the holding position of the stock: [0,100], continuous
State space: [current stock price, current holding postion, time to maturity]
Reward: profit and loss
RL algorithm: TAD3 

"""

###########################################
###############   Simulation  #############
###########################################
"""
To simulate paths for the underlying stock and European call option with constant volatility and stochastic volatility;
Constant volatility: BSM delta is calculated (benchmark)
Stochastic volatility: SABR model by Hagen et al (2002) is used. “Practitioner delta" and  “Bartlett’s delta” (Bartlett, 2006)
are calculated (benchmark)

"""

import random
import numpy as np
from scipy.stats import norm

random.seed(1)

#############################
####   Constant vol  ########
#############################

def brownian_sim(num_path, num_period, mu, std, init_p, dt):
    """
    Assume dSt = St (mu*dt + std*dWt), where Wt is brownian motion
    Input: num_path: number of path to simulate; num_period: the length of a path; init_p: initial price
    Return un_price, the underlying stock price
    """
    z = np.random.normal(size=(num_path,num_period))
    
    un_price = np.zeros((num_path,num_period))
    un_price[:,0] = init_p
    
    for t in range(num_period-1):
        un_price[:,t+1] = un_price[:,t] * np.exp((mu - (std ** 2)/ 2)* dt + std * np.sqrt(dt) * z[:,t])
    
    return un_price



def bs_call(iv, T, S, K, r, q):
    """
    BSM Call Option Pricing Formula & BS Delta formula 
    Input: T here is time to maturity, iv : implied volatility, q : continuous dividend,
            r : risk free rate, S : current stock price, K : strike price
    Return bs_price, BSM call option price;  bs_delta, BSM delta
    """
    
    d1 = (np.log(S / K) + (r - q + iv * iv / 2) * T) / (iv * np.sqrt(T))
    d2 = d1 - iv * np.sqrt(T)
    bs_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    bs_delta = np.exp(-q * T) * norm.cdf(d1)
    return bs_price, bs_delta


def get_sim_path(M, freq, np_seed, num_sim, mu=0.05, vol=0.2, S=100, K=100, r=0, q=0):                                                                
    """ 
    Simulate paths
    Input: M: initial time to maturity, days; freq: trading freq in unit of day, e.g. freq=2: every 2 day; freq=0.5 twice a day;
            np_seed: numpy random seed; num_sim: number of simulation path; mu: annual return; vol: annual volatility
            S: initial asset value; K: strike price; r: annual risk free rate; q: annual dividend
            If risk-neutrality, mu = r-q
    Return simulated data: a tuple of three arrays
        1) asset price paths (num_path x num_period)
        2) option price paths (num_path x num_period)
        3) delta (num_path x num_period)
    """
    # set the np random seed
    np.random.seed(np_seed)
    
    # Annual Trading Day
    T = 250

    # Change into unit of year
    dt = 0.004 * freq

    # Number of period
    num_period = int(M / freq)

    # underlying asset price 2-d array
    print("1. generate asset price paths")
    un_price = brownian_sim(num_sim, num_period + 1, mu, vol, S, dt)

    # time to maturity "rank 1" array: e.g. [M, M-1, ..., 0]
    ttm = np.arange(M, -freq, -freq) #np.arrage(start,stop,step) from  [start,stop)

    # BS price 2-d array and bs delta 2-d array
    print("2. generate BS price and delta")
    bs_price, bs_delta = bs_call(vol, ttm / T, un_price, K, r, q)  # bs_call(iv, T, S, K, r, q)

    print("simulation done!")

    return un_price, bs_price, bs_delta

#############################
####   Stochastic vol  ######
#############################

def sabr_sim(num_path, num_period, mu, std, init_p, dt, rho, beta, volvol):
    """
     We assume an extension of geometric Brownian motion where the volatility is stochastic : dS =µSdt+σSdz_1  ;  dσ =vσdz_2
     Input: rho: the constant correlation between dz_1 and dz_2, two Wiener processes
             volvol: the volatility of volatility process, std : initial volatility
     Return a_price, underlying asset price path; vol, the volatility path
    """
    qs = np.random.normal(size=(num_path, num_period))
    qi = np.random.normal(size=(num_path, num_period))
    qv = rho * qs + np.sqrt(1 - rho * rho) * qi   #sum of normal is normal --> construct a wiener process dz2 with correlation rho 

    vol = np.zeros((num_path, num_period))
    vol[:, 0] = std

    a_price = np.zeros((num_path, num_period))
    a_price[:, 0] = init_p

    for t in range(num_period - 1):
        gvol = vol[:, t] * (a_price[:, t] ** (beta - 1))  #beta = 1 
        a_price[:, t + 1] = a_price[:, t] * np.exp(
            (mu - (gvol ** 2) / 2) * dt + gvol * np.sqrt(dt) * qs[:, t]
        )
        vol[:, t + 1] = vol[:, t] * np.exp(
            -volvol * volvol * 0.5 * dt + volvol * qv[:, t] * np.sqrt(dt)
        )

    return a_price, vol


def sabr_implied_vol(vol, T, S, K, r, q, beta, volvol, rho):
    """ 
    Input: vol is initial volatility, T time to maturity
    Return implied volatility  SABRIV
    """

    F = S * np.exp((r - q) * T)
    x = (F * K) ** ((1 - beta) / 2)
    y = (1 - beta) * np.log(F / K)
    A = vol / (x * (1 + y * y / 24 + y * y * y * y / 1920))
    B = 1 + T * (
        ((1 - beta) ** 2) * (vol * vol) / (24 * x * x)
        + rho * beta * volvol * vol / (4 * x)
        + volvol * volvol * (2 - 3 * rho * rho) / 24
    )
    Phi = (volvol * x / vol) * np.log(F / K)
    Chi = np.log((np.sqrt(1 - 2 * rho * Phi + Phi * Phi) + Phi - rho) / (1 - rho))

    SABRIV = np.where(F == K, vol * B / (F ** (1 - beta)), A * B * Phi / Chi)

    return SABRIV


def bartlett(sigma, T, S, K, r, q, ds, beta, volvol, rho): 
    """
    Return barlett delta
    """

    dsigma = ds * volvol * rho / (S ** beta)

    vol1 = sabr_implied_vol(sigma, T, S, K, r, q, beta, volvol, rho)  #sabr_implied_vol(vol, T, S, K, r, q, beta, volvol, rho): sigma here is initial volatility
    vol2 = sabr_implied_vol(sigma + dsigma, T, S + ds, K, r, q, beta, volvol, rho)

    bs_price1, _ = bs_call(vol1, T, S, K, r, q)
    bs_price2, _ = bs_call(vol2, T, S+ds, K, r, q)

    b_delta = (bs_price2 - bs_price1) / ds

    return b_delta


def get_sim_path_sabr(M, freq, np_seed, num_sim, mu=0.05, vol=0.2, S=100, K=100, r=0, q=0, beta=1, rho=-0.4, volvol = 0.6, ds = 0.001):
    """ 
        Input: M: initial time to maturity; freq: trading freq in unit of day, e.g. freq=2: every 2 day; freq=0.5 twice a day;
            np_seed: numpy random seed; num_sim: number of simulation path; 
        Return simulated data: a tuple of four arrays
            1) asset price paths (num_path x num_period)
            2) option price paths (num_path x num_period)
            3) bs delta (num_path x num_period)
            4) bartlett delta (num_path x num_period)
    """
    # set the np random seed
    np.random.seed(np_seed)

    # Annual Trading Day
    T = 250

    # Change into unit of year
    dt = 0.004 * freq

    # Number of period
    num_period = int(M / freq)

    # asset price 2-d array; sabr_vol
    print("1. generate asset price paths (sabr)")
    a_price, sabr_vol = sabr_sim(
        num_sim, num_period + 1, mu, vol, S, dt, rho, beta, volvol
    )

    # time to maturity "rank 1" array: e.g. [M, M-1, ..., 0]
    ttm = np.arange(M, -freq, -freq)

    # BS price 2-d array and bs delta 2-d array
    print("2. generate BS price, BS delta, and Bartlett delta")

    # sabr implied vol
    implied_vol = sabr_implied_vol(
        sabr_vol, ttm / T, a_price, K, r, q, beta, volvol, rho
    )

    bs_price, bs_delta = bs_call(implied_vol, ttm / T, a_price, K, r, q)

    bartlett_delta = bartlett(sabr_vol, ttm / T, a_price, K, r, q, ds, beta, volvol, rho)

    print("simulation done!")

    return a_price, bs_price, bs_delta, bartlett_delta


###########################################
###############   memory      #############
###########################################
"""
Replay buffer

"""


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


###########################################
###############   environment      ########
###########################################
"""new env"""

"""
A new trading environment, which contains
(1) observation space: already simulated samples 
(2) action space: continuous or discrete, holdings of the hedging assets
(3) reset(): go to a new episode
(4) step(): transist to the next state

Now it also considers the intraday volumne difference --> different price impact: assume trade 5 times a day, multi_t [2,1.5,1,1.5,2]

"""

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class TradingEnv(gym.Env):
    """
    trading environment;
    contains observation space (already simulated samples), action space, reset(), step()
    """

    def __init__(self, cash_flow_flag=0, dg_random_seed=1, num_sim=500002, sabr_flag = False,
        continuous_action_flag=False, ticksize=0.1, init_ttm=10, trade_freq=0.2, num_contract=1, sim_flag = True,
        mu=0.05, vol=0.2, S=100, K=100, r=0, q=0, beta=1, rho=-0.4, volvol = 0.6, ds = 0.001, k=0):
        """ cash_flow_flag: 1 if reward is defined using cash flow, 0 if profit and loss; dg_random_seed: random seed for simulation;
            num_sim: number of paths to simulate; sabr_flag: whether use sabr model; continuous_action_flag: continuous or discrete
            action space; init_ttm: initial time to maturity in unit of day; trade_freq: trading frequency in unit of day;
            num_contract: number of call option to short; sim_flag = simulation or market data;
            Assume the trading cost include spread cost and price impact cost, cost(n)= multi_t*ticksize*(|n|+0.01*n^2)
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
        
        # time of a day [1,2,3,4,5,1,2,3,4,5.....] corresponds to [2,1.5,1,1.5,2.....]
        self.time_of_day = [2,1.5,1,1.5,2]*init_ttm

        # cost part
        self.ticksize = ticksize  
        #self.multi = multi

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
        
         #get the multi_t
        multi_t = self.time_of_day[self.t]

        # update time/period
        self.t = self.t + 1

        # get state for tomorrow
        price = self.path[self.sim_episode, self.t]
        position = action
        ttm = self.ttm_array[self.t]

        self.state = [price, position, ttm]   #state transist to next price, ttm and stores current action(position)
        
       
        
        # calculate period reward (part 1)
        cash_flow = -(position - current_position) * current_price - (np.abs(position - current_position) +0.01*(position - current_position)**2) * self.ticksize * multi_t    

        # if tomorrow is end of episode, only when at the end day done=True , self.num_period = T/frequency +1
        if self.t == self.num_period - 1:
            done = True   #you have arrived at the terminal
            # add (stock payoff + option payoff) to cash flow
            cash_flow = cash_flow + price * position - max(price - self.strike_price, 0) * self.num_contract * 100 - (position + 0.01*position**2) * self.ticksize * multi_t  
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
        
         #get the multi_t
        multi_t = self.time_of_day[self.t]

        # update time
        self.t = self.t + 1

        # get state for tomorrow (at t + 1)
        price = self.path[self.sim_episode, self.t]
        option_price = self.option_price_path[self.sim_episode, self.t]
        position = action
        ttm = self.ttm_array[self.t]

        self.state = [price, position, ttm]

        # calculate period reward (part 1)
        reward = (price - current_price) * position - (np.abs(position - current_position) +0.01*(position - current_position)**2) * self.ticksize * multi_t 

        # if tomorrow is end of episode
        if self.t == self.num_period - 1:
            done = True
            reward = reward - (max(price - self.strike_price, 0) - current_option_price) * self.num_contract * 100 - (position + 0.01*position**2) * self.ticksize * multi_t   #liquidate option and stocks
            reward = reward - self.k / 2 * (reward)**2
        else:
            done = False
            reward = reward - (option_price - current_option_price) * self.num_contract * 100
            reward = reward - self.k / 2 * (reward)**2
        # for other info later
        info = {"path_row": self.sim_episode}

        return self.state, reward, done, info



###########################################
###############   Agent      #############
###########################################
"""
TAD3

"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, BatchNormalization, Lambda
from tensorflow.keras.optimizers import Adam
import os


class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims, fc2_dims, name, chkpt_dir='tmp/tad3'):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_tad3')

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.q = Dense(1, activation=None)

    def call(self, state, action):
        q1_action_value = tf.concat([state, action], axis=1)
        #q1_action_value = BatchNormalization(q1_action_value)
        q1_action_value = self.fc1(q1_action_value) # not automatically broadcast
        #q1_action_value = BatchNormalization(q1_action_value)
        q1_action_value = self.fc2(q1_action_value)
        #q1_action_value = BatchNormalization(q1_action_value)
        
        q = self.q(q1_action_value)

        return q

class ActorNetwork(keras.Model):
    def __init__(self, fc1_dims, fc2_dims, n_actions, name, chkpt_dir='tmp/tad3'):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_tad3')
        #self.env = env

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.mu = Dense(self.n_actions, activation='sigmoid')                         #0-1


    def call(self, state):

        #state_n = BatchNormalization()(state)
        prob = self.fc1(state)
        #prob = BatchNormalization()(prob)
        prob = self.fc2(prob)
        #prob = BatchNormalization()(prob)
        
        mu = self.mu(prob)
        mu = Lambda(lambda x: x * 100)(mu)

        return mu

class Agent():
    def __init__(self, alpha, beta, input_dims, tau, env,
            gamma=0.99, update_actor_interval=2, warmup=0,
            n_actions=1, max_size=1000000, layer1_size=400,
            layer2_size=300, batch_size=100, noise=1):
        self.gamma = gamma
        self.tau = tau
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]
        self.memory = ReplayBuffer(max_size, input_dims)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval

        self.actor = ActorNetwork(layer1_size, layer2_size,
                                    n_actions=n_actions, name='actor')

        self.critic_1 = CriticNetwork(layer1_size, layer2_size, 
                                     name='critic_1')
        self.critic_2 = CriticNetwork(layer1_size, layer2_size,
                                     name='critic_2')

        self.target_actor = ActorNetwork(layer1_size, layer2_size,
                                    n_actions=n_actions, name='target_actor')
        self.target_critic_1 = CriticNetwork(layer1_size, layer2_size, 
                                     name='target_critic_1')
        self.target_critic_2 = CriticNetwork(layer1_size, layer2_size, 
                                     name='target_critic_2')

        self.actor.compile(optimizer=Adam(learning_rate=alpha), loss='mean')      
        self.critic_1.compile(optimizer=Adam(learning_rate=beta), 
                              loss='mean_squared_error')
        self.critic_2.compile(optimizer=Adam(learning_rate=beta), 
                              loss='mean_squared_error')

        self.target_actor.compile(optimizer=Adam(learning_rate=alpha), 
                                  loss='mean')
        self.target_critic_1.compile(optimizer=Adam(learning_rate=beta), 
                              loss='mean_squared_error')
        self.target_critic_2.compile(optimizer=Adam(learning_rate=beta), 
                              loss='mean_squared_error')

        self.noise = noise
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        #choose action with exploration noise
        if self.time_step < self.warmup:
            mu = np.random.normal(scale=self.noise, size=(self.n_actions,))
        else:
            state = tf.convert_to_tensor([observation], dtype=tf.float32)
            mu = self.actor(state)[0] # returns a batch size of 1, want a scalar array
        mu_prime = mu + np.random.normal(scale=self.noise)

        mu_prime = tf.clip_by_value(mu_prime, self.min_action, self.max_action)
        self.time_step += 1

        return mu_prime

    def greedy_action(self, observation):
        #choose action without exploration noise
        
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        mu = self.actor(state)[0] # returns a batch size of 1, want a scalar array
        mu = tf.clip_by_value(mu, self.min_action, self.max_action)   #action is a tensor
        return mu

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return 

        states, actions, rewards, new_states, dones = \
                self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        actions = tf.expand_dims(actions, axis=1) 
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_states, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            target_actions = self.target_actor(states_)
            target_actions = target_actions + tf.clip_by_value(np.random.normal(scale=0.2), -0.5, 0.5)

            target_actions = tf.clip_by_value(target_actions, self.min_action, 
                                          self.max_action)
        
            q1_ = self.target_critic_1(states_, target_actions)
            q2_ = self.target_critic_2(states_, target_actions)

            q1 = tf.squeeze(self.critic_1(states, actions), 1)
            q2 = tf.squeeze(self.critic_2(states, actions), 1)

            # shape is [batch_size, 1], want to collapse to [batch_size]                         #clipped double Q learning
            q1_ = tf.squeeze(q1_, 1)
            q2_ = tf.squeeze(q2_, 1)

            critic_value_ = (q1_ + q2_)/2
            # in tf2 only integer scalar arrays can be used as indices
            # and eager exection doesn't support assignment, so we can't do
            # q1_[dones] = 0.0
            target = rewards + self.gamma*critic_value_*(1-dones)
            #critic_1_loss = tf.math.reduce_mean(tf.math.square(target - q1))
            #critic_2_loss = tf.math.reduce_mean(tf.math.square(target - q2))
            critic_1_loss = keras.losses.MSE(target, q1)
            critic_2_loss = keras.losses.MSE(target, q2)


        critic_1_gradient = tape.gradient(critic_1_loss, 
                                          self.critic_1.trainable_variables)
        critic_2_gradient = tape.gradient(critic_2_loss, 
                                          self.critic_2.trainable_variables)

        self.critic_1.optimizer.apply_gradients(
                       zip(critic_1_gradient, self.critic_1.trainable_variables))
        self.critic_2.optimizer.apply_gradients(
                       zip(critic_2_gradient, self.critic_2.trainable_variables))

        
        self.learn_step_cntr += 1

        if self.learn_step_cntr % self.update_actor_iter != 0:                                     #delayed update
            return

        with tf.GradientTape() as tape:
            new_actions = self.actor(states)
            critic_1_value = self.critic_1(states, new_actions)
            actor_loss = -tf.math.reduce_mean(critic_1_value)

        actor_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(
                        zip(actor_gradient, self.actor.trainable_variables))

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i]*(1-tau))

        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic_1.weights
        for i, weight in enumerate(self.critic_1.weights):
            weights.append(weight * tau + targets[i]*(1-tau))

        self.target_critic_1.set_weights(weights)

        weights = []
        targets = self.target_critic_2.weights
        for i, weight in enumerate(self.critic_2.weights):
            weights.append(weight * tau + targets[i]*(1-tau))

        self.target_critic_2.set_weights(weights)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.critic_1.save_weights(self.critic_1.checkpoint_file)
        self.critic_2.save_weights(self.critic_2.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.target_critic_1.save_weights(self.target_critic_1.checkpoint_file)
        self.target_critic_2.save_weights(self.target_critic_2.checkpoint_file)

    def load_models(self):
        
        print('... loading models ...')
        if os.path.exists(self.actor.checkpoint_file):
            self.actor.load_weights(self.actor.checkpoint_file)
        if os.path.exists(self.critic_1.checkpoint_file):
            self.critic_1.load_weights(self.critic_1.checkpoint_file)
        if os.path.exists(self.critic_2.checkpoint_file):
            self.critic_2.load_weights(self.critic_2.checkpoint_file)
        if os.path.exists(self.target_actor.checkpoint_file):
            self.target_actor.load_weights(self.target_actor.checkpoint_file)
        if os.path.exists(self.target_critic_1.checkpoint_file):
            self.target_critic_1.load_weights(self.target_critic_1.checkpoint_file)
        if os.path.exists(self.target_critic_2.checkpoint_file):
            self.target_critic_2.load_weights(self.target_critic_2.checkpoint_file)


###########################################
#######  plots and comparison      ########
###########################################
import matplotlib.pyplot as plt
import pandas as pd

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)
    
def plot_obj(scores, figure_file):
    plt.hist(scores)
    plt.savefig(figure_file)

def test(total_episode_test, env, agent, name, delta_flag=False, bartlett_flag=False):
        """
        hedge with model: RL/BSM
        report the objectives for every 1000 episode
        save the objectives for every episode to csv 
        
        """
        print('testing...')
        
        agent.load_models()

        u_T_store = []

        for i in range(total_episode_test):
            observation = env.reset()
            done = False
            action_store = []
            reward_store = []

            while not done:

                # prepare state
                #x = np.array(observation).reshape(1, -1)

                if delta_flag:
                    action = env.delta_path[i % env.num_path, env.t] * env.num_contract * 100
                elif bartlett_flag:
                    action = env.bartlett_delta_path[i % env.num_path, env.t] * env.num_contract * 100
                else:
                    # choose action from epsilon-greedy; epsilon has been set to -1                                    #add boltzman
                    action = agent.greedy_action(observation).numpy()[0] 

                # store action to take a look
                action_store.append(action)

                # a step
                observation, reward, done, info = env.step(action)
                reward_store.append(reward)

            # get final utility at the end of episode, and store it.
            u_T = sum(reward_store)
            u_T_store.append(u_T)

            if i % 1000 == 0:
                u_T_mean = np.mean(u_T_store)
                u_T_var = np.var(u_T_store)
                path_row = info["path_row"]
                print(info)
                with np.printoptions(precision=2, suppress=True):
                    print("episode: {} | final utility Y(0): {:.2f}; so far mean and variance of final utility was {} and {}".format(i, u_T, u_T_mean, u_T_var))       
                    print("episode: {} | rewards: {}".format(i, np.array(reward_store)))
                    print("episode: {} | action taken: {}".format(i, np.array(action_store)))
                    print("episode: {} | deltas {}".format(i, env.delta_path[path_row] * 100))  
                    print("episode: {} | stock price {}".format(i, env.path[path_row]))
                    print("episode: {} | option price {}\n".format(i, env.option_price_path[path_row] * 100))
        
        upperbound = total_episode_test + 1
        epi = np.arange(1, upperbound, 1)  
        history = dict(zip(epi, u_T_store))
        #name = os.path.join('history', name)
        df = pd.DataFrame(history,index=[0])
        df.to_csv('delta', index=False, encoding='utf-8')
        
        return u_T_store

###########################################
####    train & test -constant vol      ###
###########################################
"""
Agent interact with the environment; based on memory, train the model
"""


import numpy as np
import gym
#from utils import plotLearning
import tensorflow as tf

if __name__ == '__main__':
    tf.compat.v1.enable_eager_execution()
    #tf.compat.v1.disable_eager_execution()
    num_simulation = 50000
    env = TradingEnv(num_sim = num_simulation, continuous_action_flag=True) # see tradingenv.py for more info 
    lr = 0.001
    agent = Agent(alpha=0.001, beta=0.001, input_dims=env.num_state, tau=0.005, env=env, n_actions=1)
    agent.load_models()

    scores = []
    #eps_history = []

    for i in range(num_simulation):
        #interaction
        done = False
        score = 0
        observation = env.reset()  #[price, position, ttm], price=S, position=0, ttm=init_ttm
        while not done:
            action = agent.choose_action(observation)  #action is tensor
            action = action.numpy()[0]           #change to numpy
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            observation = observation_
            agent.learn()
        #eps_history.append(agent.epsilon)
        scores.append(score)  # score for every episode

        avg_score = np.mean(scores[-100:])
        if i % 1000 == 0:
          print('episode %.2f' % i, 'score %.2f' % score, 'average_score %.2f' % avg_score)
        #        'epsilon %.2f' % agent.epsilon)

    filename = 'tad3_tf2_simple.png'
    x = [i+1 for i in range(num_simulation)]
    plot_learning_curve(x, scores, filename)
    agent.save_models()



    total_episode_test = 10000
    env_test2 = TradingEnv(continuous_action_flag=True, sabr_flag=False, dg_random_seed=2, num_sim=total_episode_test)

    delta_u = test(total_episode_test = total_episode_test, env = env_test2, agent = agent, name='delta', delta_flag=True, bartlett_flag=False)
    rl_u = test(total_episode_test = total_episode_test, env = env_test2, agent = agent, name='rl', delta_flag=False, bartlett_flag=False)
    
    plot_obj(delta_u, figure_file='delta_u')
    plot_obj(rl_u, figure_file='rl_u')
