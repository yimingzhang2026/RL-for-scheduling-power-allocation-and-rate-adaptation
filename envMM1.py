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



class wireless_env():
    
    def __init__(self,
                 N = 20,
                 K = 5,
                 M = 1,
                 f_d = 8.0,
                 T = 0.02,
                 Pmax_dBm = 100.0,
                 n0_dBm = -114.0,
                 rayleigh_var = 1.0, 
                 shadowing_dev = 8.0,
                 R = 400, 
                 min_dist = 35,
                 equal_number_for_BS = True,
                 mode = 'sumrate',
                 reset_gains = False,
                 N_neighbors = 0,
                 seed = 2021,
                 traffic_levels = 5,
                 max_rate = 2.0,
                 packet_size = 8e6,
                 bw = 10e6):
        self.N = N
        self.K = K
        self.M = M
        self.rayleigh_var = rayleigh_var
        self.shadowing_dev = shadowing_dev
        self.R = R
        self.min_dist = min_dist
        self.equal_number_for_BS = equal_number_for_BS
        
        self.correlation = special.j0(2.0*np.pi*f_d*T)
        self.Pmax_dB = Pmax_dBm - 30
        self.Pmax = np.power(10.0,(Pmax_dBm - 30)/10)
        self.noise_var = np.power(10.0,(n0_dBm - 30)/10)
        
        # Just store the current instance and one past instance of the channel
        self.H_cell2user = collections.deque([],2)
        self.H = collections.deque([],2)
        
        self.priorities = np.ones(self.N)
        self.p = self.Pmax * np.ones((self.N,self.M))
        self.spec_eff = np.zeros((self.N, self.M))
        self.total_interf = np.zeros((self.N, self.M))
        
        
        np.random.seed(seed)
        self.channel_random_state = np.random.RandomState(seed + 2021)
        self.traffic_random_state = np.random.RandomState(seed + 4042)
        
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
                                                        R = 400, 
                                                        min_dist = 35,
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
        
        self.spec_eff, self.total_interf = channel.sumrate_multi_list_clipped(self.H[-1], self.p, self.noise_var)
        self.spec_eff[0][0] = np.log2(1.0+1000) # 30dB cap
        if self.mode == 'pfs':
            self.average_spec_eff = 0.01 + np.sum(self.spec_eff, axis = 1)
            self.weights = 1.0 / self.average_spec_eff
        elif self.mode == 'traffic':
            self.create_traffic()
            self.process_traffic()
        
        
        # To determine the state function we need current channel and one past
        # channel instance. Therefore, need to call one channel_step.
        self.channel_step()
        return
        

    
    def step(self, action):
        assert action.shape == (self.N, self.M), "action shape should be (N,M)"
        self.t += 1
        self.p = action
        self.spec_eff, self.total_interf = channel.sumrate_multi_list_clipped(self.H[-1], self.p, self.noise_var)
        self.spec_eff[0][0] = np.log2(1.0+1000) # 30dB cap
        if self.mode == 'pfs':
            self.average_spec_eff = (1.0-self.beta)*self.average_spec_eff+self.beta*np.sum(self.spec_eff, axis = 1)
            self.weights = 1.0 / self.average_spec_eff
        elif self.mode == 'traffic':
            self.process_traffic()
            
        self.channel_step()
        
        
        return 

    def create_traffic(self):
        self.weights = np.zeros(self.N) # it is the queue lengths
        self.packets = [[] for i in range(self.N)]
        self.packets_t = [[] for i in range(self.N)] # store when the packet arrived
        self.arrival_rates = [self.max_rate * self.T] * self.N

            
        self.throughput = np.zeros(self.N)
        self.processed_packets_t = [[] for i in range(self.N)] # stores delays

    def process_traffic(self):
        # load traffic
        for n in range(self.N):
            num_incoming = self.traffic_random_state.poisson(self.arrival_rates[n])
            for i in range(num_incoming):
                self.packets[n].append(np.random.exponential(self.packet_size))
                self.packets_t[n].append(self.t)
        # process traffic
        for n in range(self.N):
            tmp = int(np.sum(self.spec_eff[n],axis=-1) * self.bw * self.T)
            tmp_init = tmp
            while tmp > 0 and len(self.packets[n]) > 0:
                if tmp >= self.packets[n][0]:
                    tmp -= self.packets[n][0]
                    self.processed_packets_t[n].append(1 + self.t - self.packets_t[n][0])
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
            
            