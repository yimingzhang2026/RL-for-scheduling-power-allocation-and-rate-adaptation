# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 17:51:16 2022

@author: zyimi
"""

import os
import numpy as np
import project_backend as pb
import time
import collections
import json
import DQNPerLink as DQN

import argparse

#define test scenario
parser = argparse.ArgumentParser(description='give test scenarios.')
parser.add_argument("--pre_sinr_f", default=True,  type=bool)    
parser.add_argument("--seeds", default=[1], nargs='+', type=int)             
parser.add_argument("--episode-timeslots", default=2500, type=int)   
parser.add_argument("--timeslots", default=100000, type=int)  
parser.add_argument("--mode", default="traffic")
parser.add_argument("--max-rates", default=[ 2, 5, 10, 15, 20, 25, 30, 35, 40], nargs='+', type=float)       
parser.add_argument("--reset-gains", default=True, type=bool) 
parser.add_argument("--K", default=5, type=int)   
parser.add_argument("--N", default=10, type=int)   
parser.add_argument("--M", default=1,  type=int)  
parser.add_argument("--episodes", default=[0, 1, 5, 10, 15, 20, 25, 30, 35, 40], nargs='+', type=float)     
parser.add_argument('--json-file-policy', type=str, default='dqn_total_Mbits_b128_lr001_e8',
                   help='json file for the hyperparameters')

#define policy(model learned from train)
parser.add_argument("--policy-pre_sinr_f", default=True,  type=bool) 
parser.add_argument("--policy-seeds", default=[1], nargs='+', type=int)  
parser.add_argument("--policy-episode-timeslots", default=2500, type=int)   
parser.add_argument("--policy-timeslots", default=100000, type=int)   
parser.add_argument("--policy-mode", default="traffic")
parser.add_argument("--policy-max-rate", default=[ 2,5, 10, 15, 20, 25, 30, 35,40], type=int) 
parser.add_argument("--policy-reset-gains", default=True, type=bool) 
parser.add_argument("--policy-K", default=5, type=int)  
parser.add_argument("--policy-N", default=10, type=int)  
parser.add_argument("--policy-M", default=1, type=int) 
parser.add_argument("--logs", default=True, type=bool)


args = parser.parse_args()

    
json_file_policy = args.json_file_policy

with open ('./config/policy/'+json_file_policy+'.json','r') as f:
    options_policy = json.load(f)
    
options_policy['num_actions'] = options_policy['power_actions'] * options_policy['SINR_actions']    
if not options_policy['cuda']:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#generate arriving packets    
#thining process  
max_rates = args.max_rates
largest  = 40
T = 0.02 # duration of 1 timeslot
poisson_len = args.timeslots
seed = args.seeds
np.random.seed(seed)
largest_poisson = np.random.poisson(largest*T, poisson_len)
base_poisson = largest_poisson
poissons = [ np.zeros(poisson_len) for i in range(len(max_rates))]
print(sum(largest_poisson))
for i in range(len(max_rates)-1,-1,-1):
    print("generating poisson process")  
    rate = max_rates[i]
    if i == len(max_rates)-1:
        for j in range(len(base_poisson)):
            if base_poisson[j] !=0:
                poissons[i][j] = np.random.binomial(base_poisson[j], max_rates[i]/40, 1)
        base_poisson = poissons[i]
    else:
        for j in range(len(largest_poisson)):
            if base_poisson[j] != 0:
                poissons[i][j] = np.random.binomial(base_poisson[j], max_rates[i]/max_rates[i+1], 1)
        base_poisson = poissons[i]

for i in range(len(max_rates)-1,-1,-1):
    print('total arrivals are {}'.format(sum(poissons[i]))) 
    

def initialize_p_sinr():
     p_strategy = env.Pmax * np.random.rand(env.N)
     p_strategy_current = np.array(p_strategy)
     # Initial trans_SINR is just random
     SINR_strategy = env.pre_SINRmax_db/(options_policy['SINR_actions']-2) * (np.random.randint(4,size = env.N)-1)
     SINR_strategy_current = np.array(SINR_strategy)
     
     return p_strategy,p_strategy_current,SINR_strategy,SINR_strategy_current    
    
import tensorflow as tf

#simulation starts
for idx, max_rate in enumerate(args.max_rates):
    for seed in args.seeds:
        args.seed = seed
        for policy_seed in args.seeds:
            args.policy_seed = policy_seed

            folder_name = './simulations/N%dK%dM%dseed%dtimeslots%depisode_timeslots%dmode%sreset_gains%smax_rate%dpre_sinr_f%s'%(args.N,args.K,args.M,args.seed,args.timeslots,args.episode_timeslots,args.mode,args.reset_gains,max_rate,args.pre_sinr_f)
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
    

            policy_folder_name = './simulations/N%dK%dM%dseed%dtimeslots%depisode_timeslots%dmode%sreset_gains%smax_rate%dpre_sinr_f%s'%(args.policy_N,args.policy_K,args.policy_M,args.policy_seed,args.policy_timeslots,args.policy_episode_timeslots,args.policy_mode,args.policy_reset_gains,max_rate,args.policy_pre_sinr_f)
            if not os.path.exists(policy_folder_name):
                os.makedirs(policy_folder_name)
            
                
                

            
            for ep in args.episodes:
                
            
            
                tf.reset_default_graph()
                tf.set_random_seed(100+args.seed)
                np.random.seed(args.seed)
                
                time_calculating_strategy_takes = []
                    
                
                
                import wireless_env
                env = wireless_env.env.wireless_env(N = args.N,
                                                    M = args.M,
                                                    K = args.K,
                                                    pre_SINR = args.pre_sinr_f,
                                                    mode = args.mode,
                                                    max_rate = max_rate,
                                                    max_rates = max_rates,
                                                    poissons = poissons,
                                                    reset_gains = args.reset_gains,
                                                    seed = args.seed,
                                                    duration = args.timeslots
                                                    )
                

                policy = DQN.DQN(args,options_policy,env)
                time_calculating_strategy_takes = []
                time_optimization_at_each_slot_takes = []
                
                sum_rate_distributed_policy = []
                p_strategy_all = []
                SINR_strategy_all = []
                # link_is_activated = []
                # feedback_data = []
                # feedbacks = np.zeros(args.N)
                # penalties = np.zeros(env.N)
                
                if args.logs:
                    TX_locs = []
                    RX_locs = []
                    total_packets = []
                    total_Mbits = []
                    average_wait_time = []
                    total_served = []
                    average_delay = []
                    direct_channel_gains = []
                    spec_effs = []
                    interfs = []
                
                
                with tf.Session() as sess:
                    sess.run(policy.init)
                    if options_policy["policy"] == "ddpg":
                        policy.initialize_critic_updates(sess) 
                        policy.initialize_actor_updates(sess) 
                    else:
                        policy.initialize_updates(sess) 
                    # Start iterating voer time slots
                    for sim in range (args.timeslots):
                        policy.check_memory_restart(sess,sim)       
                        policy.update_handler(sess,sim)
                        
                        # Load an instance per training episode for testing purposes.
                        if(sim %args.episode_timeslots == 0):
                            model_destination = ('%s/%s_episode%d.ckpt'%(
                                    policy_folder_name,json_file_policy,ep)).replace('[','').replace(']','')
                            policy.load(sess,model_destination)
                            

                            env.reset()
                            env_util = wireless_env.utils2.state_and_reward(env, reward_mode=options_policy['reward_mode'])
    
                            if args.logs:
                                TX_locs.append(env.TX_loc)
                                RX_locs.append(env.RX_loc)
                            
                            
                        # If at least one time slot passed to get experience
                        if (sim %args.episode_timeslots > 1):                    
                            # Each agent picks its strategy.
                            state = env_util.get_state()
                            reward = env_util.get_reward()
                            for agent in range (env.N):
                                current_local_state = state[agent,0,:]
                                a_time = time.time()  
                                strategy = policy.act_noepsilon(sess,current_local_state,sim)
                                time_calculating_strategy_takes.append(time.time()-a_time)
                                # if sim > 50:
                                #     reward[agent] -= penalties[agent]
                                        
                                # Pick the action
  
                                p_strategy[agent] = policy.strategy_translation[strategy][0] #** 10
                                SINR_strategy[agent] = policy.strategy_translation[strategy][1]
                                # Add current state to the short term memory to observe it during the next state
                                policy.previous_state[agent,:] = current_local_state
                                policy.previous_action[agent] = strategy
                
                        if(sim %args.episode_timeslots < 1):
                            p_strategy,p_strategy_current,SINR_strategy,SINR_strategy_current = initialize_p_sinr()
                            
                        # penalties = np.zeros(env.N)
                        for n in range(env.N):
                            # if len(env.packets[n]) == 0 and p_strategy[n] > 0:
                            #     penalties[n] += 1000#p_strategy[n]
                            # penalties[n] = 0
                            # if len(env.packets[n]) > sum([len(env.packets[neigh]) for neigh in env.link_neighbors[n]]) and p_strategy[n] != 0.95 * env.Pmax:
                            #     penalties[n] += 10#p_strategy[n]
                            #     p_strategy[n] = env.Pmax
                                # SINR_strategy[n] = env.pre_SINRmax_db 
                                # if options_policy["policy"] == "ddpg":
                                #     policy.previous_action[n] = 1.0
                                # else:
                                #     policy.previous_action[n] = policy.num_actions - 1 #set to be the largest power?
                            
                                    
                            if len(env.packets[n]) == 0:
                                p_strategy[n] = 0
                                SINR_strategy[n] = -10  
                                policy.previous_action[n] = 0.0
                                    
                            # elif len(env.packets[n]) < 1/4 * max([len(env.packets[neigh]) for neigh in env.link_neighbors[n]]):
                            #     p_strategy[n] = 0
                            #     SINR_strategy[n] = -10
                            #     if options_policy["policy"] == "ddpg":
                            #         policy.previous_action[n] = 0.0
                            #     else:
                            #         policy.previous_action[n] = 0.0 
                        for k in range(env.K):
                            if sum(p_strategy[env.user_mapping[k]]) > env.Pmax:
                                # penalties[env.user_mapping[k]] = sum(p_strategy[env.user_mapping[k]]) - env.Pmax
                                p_strategy[env.user_mapping[k]] = p_strategy[env.user_mapping[k]] / sum(p_strategy[env.user_mapping[k]])
                            # if sum(p_strategy[env.user_mapping[k]]) + sum(p_strategy[env.neighbors[k]]) < 0.01:
                            #     penalties[env.user_mapping[k]] = 10.0
                                
                        p_strategy_current = np.array(p_strategy)
                        
                        sum_rate_distributed_policy.append(np.array(env.throughput))
                        SINR_strategy_all.append(np.array(SINR_strategy))                
                        p_strategy_all.append(np.array(p_strategy))
                        
                        if args.logs:
                            total_packets.append([len(ar) for ar in env.packets])
                            total_Mbits.append([sum(ar)/1e6 for ar in env.packets])
                            average_wait_time.append([env.t - sum(ar)/len(ar) if len(ar) > 0 else 0 for ar in env.packets_t])
                            total_served.append([len(ar) for ar in env.processed_packets_t])
                            average_delay.append([sum(ar)/len(ar) if len(ar) > 0 else 0 for ar in env.processed_packets_t])
                            direct_channel_gains.append(env.H[-1].diagonal())
                            spec_effs.append(np.sum(env.spec_eff,axis=1))
                            interfs.append(np.array(env.total_interf))
                        
                        if sum(p_strategy) > env.N * env.Pmax * 0.95:
                            print('all 1 sim %d'%(sim))
                        if sum(p_strategy) < env.N * env.Pmax * 0.01 and sum([len(arr) for arr in env.packets])!=0:
                            print('all 0 sim %d'%(sim))
                        if sum([len(arr) for arr in env.packets])==0:
                            print('clean queues sim %d'%(sim))
                        
                        env.step(p_strategy.reshape(env.N,env.M),SINR_strategy.reshape(env.N,env.M))
                        #feedbacks = env.feedbacks #number of countinuous failed transmissions
                        if(sim % args.episode_timeslots == 0):
                            print('Timeslot %d'%(sim))
                   
                    policy.equalize(sess)
                    print('Train is over')
                
                       
                # End Train Phase
                # np_save_path = '%s/%s%s.npz'%(folder_name,policy_folder_name.split('./simulations/')[1],json_file_policy)
                np_save_path = '%s/%s%de%dr%d.npz'%(folder_name,json_file_policy,args.policy_seed,ep,max_rate)
                print(np_save_path)
                if not args.logs:
                    np.savez(np_save_path,options_policy,sum_rate_distributed_policy,p_strategy_all,SINR_strategy_all,
                             env.processed_packets_t,time_calculating_strategy_takes)
                else:
                    np.savez(np_save_path,
                             options_policy,
                             sum_rate_distributed_policy,
                             p_strategy_all,
                             env.processed_packets_t,
                             time_calculating_strategy_takes,
                             TX_locs,
                             RX_locs,
                             total_packets,
                             total_Mbits,
                             average_wait_time,
                             total_served,
                             average_delay,
                             direct_channel_gains,
                             spec_effs,
                             interfs)