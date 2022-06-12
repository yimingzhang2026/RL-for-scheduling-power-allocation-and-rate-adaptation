# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 21:50:20 2021

@author: sinan
"""

import numpy as np

class state_and_reward():
    
    def __init__(self, env, state_mode='aggregates', reward_mode='sum', scale_R_inner = 0.75, scale_R_interf = 2.5):
        if state_mode not in ['jsac', 'aggregates']:
            raise Exception(NotImplemented)
        if reward_mode not in ['externalities', 'sum', 'sumOver10', 'total_Mbytes', 'total_MbytesOver25', 'total_Mbits', 'total_packets']:
            raise Exception(NotImplemented)
            
        self.state_mode = state_mode
        self.reward_mode = reward_mode
        
        print('tttesstt2')
        
        self.env = env
        if state_mode == 'aggregates':
            if env.mode == 'sumrate':
                self.state_dim =  6 + 4 * env.N_neighbors
            else:
                self.state_dim = 7 + 5 * env.N_neighbors + 1 #plus feedbacks
            
        scale_g_dB_R = scale_R_inner*env.R
        rb = 200.0
        if(scale_g_dB_R < rb):
            scale_g_dB = - (128.1 + 37.6* np.log10(0.001 * scale_g_dB_R))
        else:
            scale_g_dB = - (128.1 + 37.6* np.log10(scale_g_dB_R/rb) + 37.6* np.log10(0.001*rb)) 
        self.scale_gain = np.power(10.0,scale_g_dB/10.0)
        self.input_placer = np.log10(env.noise_var/self.scale_gain)
        scale_g_dB_inter_R = scale_R_interf * env.R
        if(scale_g_dB_R < rb):
            scale_g_dB_interf = - (128.1 + 37.6* np.log10(0.001 * scale_g_dB_inter_R))
        else:
            scale_g_dB_interf = - (128.1 + 37.6* np.log10(scale_g_dB_inter_R/rb) + 37.6* np.log10(0.001*rb))
        self.scale_gain_interf = np.power(10.0,scale_g_dB_interf/10.0)
        
    def get_state(self):
        if self.state_mode == 'aggregates':
            return self._aggregates_state()
    
    def get_reward(self):
        if self.reward_mode == 'sum':
            return self._sum_reward()
        elif self.reward_mode == 'sumOver10':
            return self._sum_rewardOverTen()
        elif self.reward_mode == 'externalities':
            return self._externalities()
        elif self.reward_mode == 'total_Mbytes':
            return self._total_Mbytes()
        elif self.reward_mode == 'total_MbytesOver25':
            return self._total_MbytesOver25()
        elif self.reward_mode == 'total_Mbits':
            return self._total_MbytesOver25()
        elif self.reward_mode == 'total_packets':
            return self._total_packets()
        
    def _sum_reward(self):
        reward = np.zeros((self.env.N,self.env.M))
        
        for n in range(self.env.N):
            for m in range(self.env.M):
                this_reward = 0.0
                # for n in self.env.user_mapping[k]:
                this_reward += self.env.weights[n] * self.env.spec_eff[n,m]
                for neigh in self.env.link_neighbors[n]:
                    this_reward += self.env.weights[neigh] * self.env.spec_eff[neigh,m]
                reward[n,m] = this_reward
        
        # for n in range(self.env.N):
        #     for m in range(self.env.M):
        #         this_reward = 0.0
        #         if self.env.p[n,m] == 0: 
        #             reward[n,m] = this_reward
        #         else:
        #             # for n in self.env.user_mapping[k]:
        #             this_reward += self.env.weights[n] * self.env.spec_eff[n,m]
        #             for neigh in self.env.neighbors[self.env.cell_mapping[n]]:
        #                 if self.env.p[neigh,m] != 0:
        #                     this_reward += self.env.weights[neigh] * self.env.spec_eff[neigh,m]
        #                     this_reward -= self.env.weights[neigh] * np.log2(1.0 + self.env.p[neigh,m] * (self.env.H[-2][neigh,neigh,m] **2) / (1e-20 + self.env.total_interf[neigh] - self.env.p[n,m] * (self.env.H[-2][neigh,n,m] ** 2)))
        #             for neigh in self.env.user_mapping[self.env.cell_mapping[n]]:
        #                 if neigh == n: continue
        #                 if self.env.p[neigh,m] != 0:
        #                     this_reward += self.env.weights[neigh] * self.env.spec_eff[neigh,m]
        #                     this_reward -= self.env.weights[neigh] * np.log2(1.0 + self.env.p[neigh,m] * (self.env.H[-2][neigh,neigh,m] **2) / (1e-20 + self.env.total_interf[neigh] - self.env.p[n,m] * (self.env.H[-2][neigh,n,m] ** 2)))

        #             reward[n,m] = this_reward
                
        return reward
    
    def _externalities(self):
        reward = np.zeros((self.env.N,self.env.M))
        
        for n in range(self.env.N):
            for m in range(self.env.M):
                this_reward = 0.0
                if self.env.p[n,m] == 0: 
                    reward[n,m] = this_reward
                else:
                    # for n in self.env.user_mapping[k]:
                    this_reward += self.env.weights[n] * self.env.spec_eff[n,m]
                    for neigh in self.env.neighbors[self.env.cell_mapping[n]]:
                        if self.env.p[neigh,m] != 0:
                            this_reward += self.env.weights[neigh] * self.env.spec_eff[neigh,m]
                            this_reward -= self.env.weights[neigh] * np.log2(1.0 + self.env.p[neigh,m] * (self.env.H[-2][neigh,neigh,m] **2) / (1e-20 + self.env.total_interf[neigh] - self.env.p[n,m] * (self.env.H[-2][neigh,n,m] ** 2)))

                    reward[n,m] = this_reward
                
        return reward
    
    def _sum_rewardOverTen(self):
        reward = np.zeros((self.env.N,self.env.M))
        
        for n in range(self.env.N):
            for m in range(self.env.M):
                this_reward = 0.0
                # for n in self.env.user_mapping[k]:
                this_reward += self.env.weights[n] * self.env.spec_eff[n,m]
                for neigh in self.env.link_neighbors[n]:
                    this_reward += self.env.weights[neigh] * self.env.spec_eff[neigh,m]
                reward[n,m] = this_reward
                
        return reward / 10.0

    def _total_Mbytes(self):
        reward = np.zeros((self.env.N,self.env.M))
        
        for n in range(self.env.N):
            for m in range(self.env.M):
                this_reward = 0.0
                # for n in self.env.user_mapping[k]:
                this_reward -= sum(self.env.packets[n]) / 8e6
                for neigh in self.env.link_neighbors[n]:
                    this_reward -= sum(self.env.packets[neigh]) / 8e6
                reward[n,m] = this_reward
                
        return reward / 10.0

    def _total_Mbits(self):
        reward = np.zeros((self.env.N,self.env.M))
        
        for n in range(self.env.N):
            for m in range(self.env.M):
                this_reward = 0.0
                # for n in self.env.user_mapping[k]:
                this_reward -= sum(self.env.packets[n]) / 1e6
                for neigh in self.env.link_neighbors[n]:
                    this_reward -= sum(self.env.packets[neigh]) / 1e6
                reward[n,m] = this_reward
                
        return reward / self.env.max_rate
    
    def _total_MbytesOver25(self):
        reward = np.zeros((self.env.N,self.env.M))
        
        for n in range(self.env.N):
            for m in range(self.env.M):
                this_reward = 0.0
                # for n in self.env.user_mapping[k]:
                this_reward -= sum(self.env.packets[n]) / 8e6
                for neigh in self.env.link_neighbors[n]:
                    this_reward -= sum(self.env.packets[neigh]) / 8e6
                reward[n,m] = this_reward
                
        return reward / 25.0
    
    def _total_packets(self):
        reward = np.zeros((self.env.N,self.env.M))
        
        for n in range(self.env.N):
            for m in range(self.env.M):
                this_reward = 0.0
                # for n in self.env.user_mapping[k]:
                this_reward -= sum(self.env.packets[n]) / self.env.packet_size
                for neigh in self.env.link_neighbors[n]:
                    this_reward -= min(20,sum(self.env.packets[neigh]) / self.env.packet_size)
                reward[n,m] = this_reward
                
        return reward / self.env.max_rate
        
    def _aggregates_state(self):
        state = np.zeros((self.env.N, self.env.M, self.state_dim))
        
        
        for n in range(self.env.N):
            for m in range(self.env.M):
                cursor = 0
                k = self.env.cell_mapping[n]
                state[n, m, cursor] = self.env.p[n,m] / self.env.Pmax
                cursor += 1
                if self.env.mode == 'pfs':
                    state[n, m, cursor] = 1.0 / self.env.weights[n]
                    cursor += 1
                elif self.env.mode == 'traffic':
                    state[n, m, cursor] = len(self.env.packets[n])
                    cursor += 1
                state[n, m, cursor] = np.log10((self.env.H[-1][n,n,m] ** 2)/self.scale_gain)
                cursor += 1
                state[n, m, cursor] = np.log10((self.env.H[-2][n,n,m] ** 2)/self.scale_gain)
                cursor += 1
                state[n, m, cursor] = 0.5 * self.env.spec_eff[n, m]
                cursor += 1
                state[n, m, cursor] = np.log10((self.env.noise_var+np.matmul(np.delete(self.env.H[-1][n,:,m] ** 2,n),
                                       np.delete(self.env.p[:,m],n)))/self.scale_gain)
                cursor += 1
                if self.env.total_interf[n,m] == self.env.noise_var:
                    state[n, m, cursor] = self.input_placer
                else:
                    state[n, m, cursor] = np.log10(self.env.total_interf[n,m]/self.scale_gain)
                cursor += 1
                state[n, m, cursor] = self.env.feedbacks[n,m]
                cursor += 1
                
            for neigh in self.env.link_neighbors[n]:
                state[n, m, cursor] = np.log10(self.env.gains[neigh,n]/self.scale_gain)
                cursor += 1
                if self.env.mode == 'pfs':
                    state[n, m, cursor] = 1.0 / self.env.weights[neigh]
                    cursor += 1
                elif self.env.mode == 'traffic':
                    state[n, m, cursor] = len(self.env.packets[neigh])
                    cursor += 1
                state[n, m, cursor] = 0.5 * self.env.spec_eff[neigh, m]
                cursor += 1
                state[n, m, cursor] = np.log10((self.env.H[-2][neigh,neigh,m] ** 2)/self.scale_gain)
                cursor += 1
                if self.env.total_interf[neigh,m] == self.env.noise_var:
                    state[n, m, cursor] = self.input_placer
                else:
                    state[n, m, cursor] = np.log10(self.env.total_interf[neigh,m]/self.scale_gain)
                cursor += 1
                
        return state
                
                
                
                
                
                
                
                
                
                
                