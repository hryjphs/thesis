# -*- coding: utf-8 -*-
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