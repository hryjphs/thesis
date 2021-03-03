# -*- coding: utf-8 -*-
"""
Agent interact with the environmen; based on memory, train the model
"""

from simpleDQNagent import Agent
import numpy as np
import gym
#from utils import plotLearning
import tensorflow as tf
from tradingenv import TradingEnv
from visual import plotLearning

if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    num_simulation = 50000
    env = TradingEnv(num_sim = num_simulation) # see tradingenv.py for more info 
    lr = 0.001
    agent = Agent(gamma=0.99, epsilon=1.0, lr=lr, 
                input_dims=env.num_state,
                mem_size=1000, batch_size=64,
                epsilon_end=0.01, env=env)
    

    scores = []
    eps_history = []

    for i in range(num_simulation):
        #interaction
        done = False
        score = 0
        observation = env.reset()  #[price, position, ttm], price=S, position=0, ttm=init_ttm
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            observation = observation_
            agent.learn()
        eps_history.append(agent.epsilon)
        scores.append(score)  # score for every episode

        avg_score = np.mean(scores[-100:])
        print('episode: ', i, 'score %.2f' % score,
                'average_score %.2f' % avg_score,
                'epsilon %.2f' % agent.epsilon)

    filename = 'dqn_tf2.png'
    x = [i+1 for i in range(num_simulation)]
    plotLearning(x, scores, eps_history, filename)