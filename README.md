# thesis
### 2021/3/3: 
Simulation.py includes BSM and SABR ; tradingenv.py defines the basic trading environment with fritions; memory.py includes Replaybuffer; simpleDQNagent.py defines a basic DQN agent; visual.py includes a basic plot presenting the moving average of the objective; train.py starts to train a baseline DQN model.  

This is a very preliminary project. For example, agents using other RL algorithms should be added / Prioritized replaybuffer ? / market data processor? / more plots? 

### 2021/4/3: 
In TAD3 file: train.py :

Initial time to maturity is 10 days; Assume we could trade 5 times a day; short 1 European call option contract ( each for 100 shares)
Geometric brownian simulation (constant volatility) or SABR (stochastic volatility) to simulate the underlying stock prices
Action space: the holding position of the stock: [0,100], continuous
State space: [current stock price, current holding postion, time to maturity]
Reward: profit and loss
RL algorithm: TAD3 


### 2021/4/9: 
In DDDQN_PRE file: 

Initial time to maturity is 10 days; Assume we could trade 5 times a day; short 1 European call option contract ( each for 100 shares)
Geometric brownian simulation (constant volatility) or SABR (stochastic volatility) to simulate the underlying stock prices
Action space: the holding position of the stock: [0,100], DISCRETE
State space: [current stock price, current holding postion, time to maturity]
Reward: profit and loss
RL algorithm: Double dueling DQN with prioritized  replay  experience-->stablize, avoid overestimation, speed up 
