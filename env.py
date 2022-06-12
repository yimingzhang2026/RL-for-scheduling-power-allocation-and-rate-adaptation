# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 17:11:12 2021

@author: sinan
"""

import numpy as np
import collections
import wireless_env.channel as channel
from scipy import special
import math

global  max_rates 
global  poissons

class wireless_env():
    
    def __init__(self,
                 N = 20,
                 K = 5,
                 M = 1,
                 f_d = 10.0,
                 T = 0.02,
                 Pmax_dBm = 23.0,
                 n0_dBm = -114.0,
                 rayleigh_var = 1.0, 
                 shadowing_dev = 8.0,
                 R = 500, 
                 min_dist = 35,
                 equal_number_for_BS = True,
                 mode = 'sumrate',
                 reset_gains = False,
                 N_neighbors = 4,
                 seed = 2021,
                 traffic_levels = 5,
                 max_rate = 2.0,
                 packet_size = 5e5,
                 bw = 10e6,
                 traffic_seed = None,
                 pre_SINR = False,
                 pre_SINRmax_db = 30,
                 duration = 100000,
                 max_rates = [],
                 poissons = []):
        

        
        
        self.max_rates = max_rates
        self.poissons = poissons
        self.seed = seed
        
        if traffic_seed is None: self.traffic_seed = self.seed
        self.N = N
        self.K = K
        self.M = M
        self.rayleigh_var = rayleigh_var
        self.shadowing_dev = shadowing_dev
        self.R = R
        self.min_dist = min_dist
        self.equal_number_for_BS = equal_number_for_BS
        
        self.link_activation = np.zeros(N)
        self.feedbacks = np.zeros((self.N,self.M))
        
        self.correlation = special.j0(2.0*np.pi*f_d*T)
        self.Pmax_dB = Pmax_dBm - 30
        self.Pmax = np.power(10.0,(Pmax_dBm - 30)/10)
        self.noise_var = np.power(10.0,(n0_dBm - 30)/10)
        
        self.pre_SINR = pre_SINR
        self.pre_SINRmax_db = pre_SINRmax_db
        self.PRE_SINR = np.zeros(self.N)
        self.SINR = np.zeros((self.N, self.M))
        
        # Just store the current instance and one past instance of the channel
        self.H_cell2user = collections.deque([],2)
        self.H = collections.deque([],2)
        
        self.priorities = np.ones(self.N)
        self.p = self.Pmax * np.ones((self.N,self.M))
        self.spec_eff = np.zeros((self.N, self.M))
        self.total_interf = np.zeros((self.N, self.M))
        
        
        np.random.seed(self.seed)
        self.channel_random_state = np.random.RandomState(self.seed + 2021)
        self.traffic_random_state = np.random.RandomState(self.traffic_seed + 4042)
        
        self.duration = duration


        
        self.mode = mode
        if self.mode == 'sumrate':
            self.weights = np.ones(self.N)
        elif self.mode == 'pfs':
            self.beta = 0.01
            self.average_spec_eff = np.zeros(self.N)
            self.weights = np.zeros(self.N)
        elif self.mode == 'traffic':
            self.bw = bw
            self.packet_size = packet_size
            self.max_rate = max_rate
            self.traffic_levels = traffic_levels
            self.T = T
        
            self.create_traffic()
            
        self.reset_gains = reset_gains
        
        self.t = -1
        

        
        self.N_neighbors = N_neighbors
        self.neighbors = np.zeros((self.K, self.N_neighbors)).astype(np.int)
        self.link_neighbors = np.zeros((self.N, self.N_neighbors)).astype(np.int)
        
        
        
        

        
    def channel_step(self):
        self.state_cell2user = channel.get_markov_rayleigh_variable(
                                state = self.state_cell2user,
                                correlation = self.correlation,
                                N = self.N,
                                random_state = self.channel_random_state,
                                M = self.M, 
                                K = self.K)
        self.H_cell2user.append(np.multiply(np.sqrt(self.gains_cell2user), abs(self.state_cell2user)))
        tmp_H = np.zeros((self.N,self.N,self.M))
        for n in range(self.N):
            tmp_H[n,:,:] = np.multiply(np.sqrt(self.gains[n,:,:]), abs(self.state_cell2user[n,self.cell_mapping,:]))
        self.H.append(tmp_H)
        
        
    def reset(self):
        if self.t == -1 or self.reset_gains:
            channel_parameters = channel.generate_Cellular_CSI(N = self.N, 
                                                        K = self.K, 
                                                        random_state = self.channel_random_state,
                                                        M = self.M, 
                                                        rayleigh_var = 1.0, 
                                                        shadowing_dev = 8.0,
                                                        R = self.R, 
                                                        min_dist = self.min_dist,
                                                        equal_number_for_BS = True)
            
            self.gains, self.gains_cell2user, self.cell_mapping, self.user_mapping, self.TX_loc, self.RX_loc = channel_parameters

            # Get neighbors with respect to large scale fading.
            for k in range(self.K):
                tmp = np.argsort(self.gains_cell2user[:,k].reshape(self.N))[::-1]
                tmp = np.delete(tmp, np.where((tmp[:, None] == self.user_mapping[k]).any(axis=1)))
                assert tmp.shape[0] >= self.N_neighbors, 'Not enough neighbors, consider reducing N_neighbors'
                self.neighbors[k] = tmp[:self.N_neighbors]
            for n in range(self.N):
                tmp = np.argsort(self.gains[:,n].reshape(self.N))[::-1]
                tmp = np.delete(tmp, np.where((tmp[:, None] == n).any(axis=1)))
                assert tmp.shape[0] >= self.N_neighbors, 'Not enough neighbors, consider reducing N_neighbors'
                self.link_neighbors[n] = tmp[:self.N_neighbors]
            # Compute rayleigh fading for time slot = 0
            self.state_cell2user = channel.get_random_rayleigh_variable(N = self.N, 
                                                                        random_state = self.channel_random_state,
                                                                        M = self.M, 
                                                                        K = self.K, 
                                                                        rayleigh_var = 1.0)
            self.H_cell2user.append(np.multiply(np.sqrt(self.gains_cell2user), abs(self.state_cell2user)))
            tmp_H = np.zeros((self.N,self.N,self.M))
            for n in range(self.N):
                tmp_H[n,:,:] = np.multiply(np.sqrt(self.gains[n,:,:]), abs(self.state_cell2user[n,self.cell_mapping,:]))
            self.H.append(tmp_H)
        
        self.t = 0
        
        self.SINR, self.spec_eff, self.total_interf = channel.sumrate_multi_list_clipped(self.H[-1], self.p, self.noise_var)
        if self.mode == 'pfs':
            self.average_spec_eff = 0.01 + np.sum(self.spec_eff, axis = 1)
            self.weights = 1.0 / self.average_spec_eff
        elif self.mode == 'traffic':
            self.create_traffic()
        
        
        # To determine the state function we need current channel and one past
        # channel instance. Therefore, need to call one channel_step.
        self.channel_step()
        if self.mode == 'traffic':
            self.load_traffic()
            
        return
        

    
    def step(self, p_action, SINR_action):
        assert p_action.shape == (self.N, self.M), "action shape should be (N,M)"
        assert SINR_action.shape == (self.N, self.M), "action shape should be (N,M)"
        self.t += 1
        self.p = p_action
        self.PRE_SINR = SINR_action
        self.SINR, self.spec_eff, self.total_interf = channel.sumrate_multi_list_clipped(self.H[-1], self.p, self.noise_var)
        if self.mode == 'pfs':
            self.average_spec_eff = (1.0-self.beta)*self.average_spec_eff+self.beta*np.sum(self.spec_eff, axis = 1)
            self.weights = 1.0 / self.average_spec_eff
        elif self.mode == 'traffic':
            self.process_traffic()
            
        self.channel_step()
        
        if self.mode == 'traffic':
            self.load_traffic()
        
        return 
    
    # def stepwmmse(self, action):
    #     assert action.shape == (self.N, self.M), "action shape should be (N,M)"
    #     self.t += 1
    #     self.p = action
    #     self.SINR, self.spec_eff, self.total_interf = channel.sumrate_multi_list_clipped(self.H[-1], self.p, self.noise_var)
    #     if self.mode == 'pfs':
    #         self.average_spec_eff = (1.0-self.beta)*self.average_spec_eff+self.beta*np.sum(self.spec_eff, axis = 1)
    #         self.weights = 1.0 / self.average_spec_eff
    #     elif self.mode == 'traffic':
    #         self.process_traffic_wmmse()
            
    #     self.channel_step()
        
    #     if self.mode == 'traffic':
    #         self.load_traffic()
        
    #     return 

    def create_traffic(self):
        self.weights = np.zeros(self.N) # it is the queue lengths
        self.packets = [[] for i in range(self.N)]
        self.packets_t = [[] for i in range(self.N)] # store when the packet arrived
        np.random.seed(self.seed)
        self.arrival_rates = self.max_rate / self.traffic_levels * self.traffic_random_state.choice(range(1, 1+self.traffic_levels),size=self.N) 
#        self.arrival_rates = self.max_rate / self.traffic_levels * self.traffic_random_state.choice(range(1, 1+self.traffic_levels),size=self.N)*self.T
        self.poisson_process = [ np.zeros(self.duration) for i in range(self.N)]
        index = self.max_rates.index(self.max_rate)
        base_poisson = self.poissons[index]
        
        for n in range(self.N):
            for j in range(len(base_poisson)):
                if base_poisson[j] !=0:
                    self.poisson_process[n][j] = np.random.binomial(base_poisson[j], self.arrival_rates[n]/self.max_rate, 1)
        
        # for n in range(self.N):
        #     size = self.duration
        #     np.random.seed(self.seed)
        #     self.poisson_process[n] = self.traffic_random_state.poisson(self.arrival_rates[n], self.duration)
        # # self.arrival_rates = self.max_rate * self.T * np.ones(self.N)
            
        self.throughput = np.zeros(self.N)
        self.processed_packets_t = [[] for i in range(self.N)] # stores delays
        
    def load_traffic(self):
        # load traffic
        for n in range(self.N):
            num_incoming = int(self.poisson_process[n][self.t])
            for i in range(num_incoming):
                # self.packets[n].append(self.traffic_random_state.exponential(self.packet_size))
                self.packets[n].append(self.packet_size)
                self.packets_t[n].append(self.t)
                
    def process_traffic(self):
        # process traffic
        for n in range(self.N):       
            if np.sum(self.PRE_SINR[n],axis=-1) == -10:
                self.spec_eff[n] = 0 
                self.link_activation[n] = 0
                self.feedbacks[n][0] = 0
            elif np.sum(self.PRE_SINR[n],axis=-1) > self.SINR[n]:
                self.spec_eff[n][0] = 0 
                self.link_activation[n] = 0
                self.feedbacks[n][0] += 1
            else :
                ratio = np.power(10, np.sum(self.PRE_SINR[n],axis=-1)/10.0)
                self.spec_eff[n] = np.log2(1 + ratio)
                self.link_activation[n] = 1
                self.feedbacks[n] = 0
                
            tmp = int(np.sum(self.spec_eff[n],axis=-1) * self.bw * self.T)
            tmp_init = tmp
            while tmp > 0 and len(self.packets[n]) > 0:
                if tmp >= self.packets[n][0]:
                    tmp -= self.packets[n][0]
                    # check the correctness of this calculation.
                    self.processed_packets_t[n].append(self.t - self.packets_t[n][0])
                    del(self.packets[n][0])
                    del(self.packets_t[n][0])
                else:
                    self.packets[n][0] -= tmp
                    tmp = 0
            # while len(self.packets[n]) > 0:
            #     if self.t - self.packets_t[n][0] > 500:
            #         self.processed_packets_t[n].append(500)
            #         del(self.packets[n][0])
            #         del(self.packets_t[n][0])
            #     else: 
            #         break
            self.throughput[n] = tmp_init - tmp
            
    # def process_traffic_wmmse(self):
    #     # process traffic
    #     for n in range(self.N):
                                
    #         tmp = int(np.sum(self.spec_eff[n],axis=-1) * self.bw * self.T)
    #         tmp_init = tmp
    #         while tmp > 0 and len(self.packets[n]) > 0:
    #             if tmp >= self.packets[n][0]:
    #                 tmp -= self.packets[n][0]
    #                 # check the correctness of this calculation.
    #                 self.processed_packets_t[n].append(self.t - self.packets_t[n][0])
    #                 del(self.packets[n][0])
    #                 del(self.packets_t[n][0])
    #             else:
    #                 self.packets[n][0] -= tmp
    #                 tmp = 0
    #         # while len(self.packets[n]) > 0:
    #         #     if self.t - self.packets_t[n][0] > 500:
    #         #         self.processed_packets_t[n].append(500)
    #         #         del(self.packets[n][0])
    #         #         del(self.packets_t[n][0])
    #         #     else: 
    #         #         break
    #         self.throughput[n] = tmp_init - tmp
        