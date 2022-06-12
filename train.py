# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 16:55:27 2022

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

parser = argparse.ArgumentParser(description='give test scenarios.')
parser.add_argument("--seeds", default=[2022], nargs='+', type=int)              
parser.add_argument("--episode-timeslots", default=2500, type=int)   
parser.add_argument("--timeslots", default=100000, type=int)   
parser.add_argument("--mode", default="traffic")
parser.add_argument("--max-rates", default=[ 2, 5, 10, 15, 20, 25, 30,35,40], nargs='+', type=float)
parser.add_argument("--reset-gains", default=True, type=bool) 
parser.add_argument("--K", default=5, type=int)   
parser.add_argument("--N", default=10, type=int) 
parser.add_argument("--M", default=1,  type=int)   
parser.add_argument("--pre_sinr_f", default=True,  type=bool)     
parser.add_argument('--json-file-policy', type=str, default='dqn_total_Mbits_b128_lr001_e8',
                   help='json file for the hyperparameters')
args = parser.parse_args()
json_file_policy = args.json_file_policy
with open ('./config/policy/'+json_file_policy+'.json','r') as f:
    options_policy = json.load(f)   
options_policy['num_actions'] = options_policy['power_actions'] * options_policy['SINR_actions']
if not options_policy['cuda']:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf


#thining process  
#generate basic poisson process with largest max_rate
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



    
#make simulation folder           
if not os.path.exists("./simulations"):
    os.makedirs("./simulations")
    
    
    
#generate random power and data rate for 1st timeslot    
def initialize_p_sinr():
    p_strategy = env.Pmax * np.random.rand(env.N)
    p_strategy_current = np.array(p_strategy)
    # Initial trans_SINR is just random
    np.random.seed(seed)
    SINR_strategy = env.pre_SINRmax_db/(options_policy['SINR_actions']-2) * (np.random.randint(4,size = env.N)-1)
    SINR_strategy_current = np.array(SINR_strategy)
    
    return p_strategy,p_strategy_current,SINR_strategy,SINR_strategy_current  


#train for different packets load and random seed
for idx, max_rate in enumerate(args.max_rates):
    for seed in args.seeds:
        args.seed = seed
        folder_name = './simulations/N%dK%dM%dseed%dtimeslots%depisode_timeslots%dmode%sreset_gains%smax_rate%dpre_sinr_f%s'%(args.N,args.K,args.M,args.seed,args.timeslots,args.episode_timeslots,args.mode,args.reset_gains,max_rate,args.pre_sinr_f)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        
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
        
        #initialization
        time_calculating_strategy_takes = []
        time_optimization_at_each_slot_takes = []
        p_strategy_all = []
        SINR_strategy_all = []
#        penalties = np.zeros(env.N)
        old_reward = np.zeros(env.N)
        

        
        #record data
        sum_rate_distributed_policy = []
        link_is_activated = []        
        # feedbacks = np.zeros(args.N)
        
        
        with tf.Session() as sess:
            sess.run(policy.init)
            if options_policy["policy"] == "ddpg":
                policy.initialize_critic_updates(sess) 
                policy.initialize_actor_updates(sess) 
            else:
                policy.initialize_updates(sess) 
            # Start iterating over timeslots
            for sim in range (args.timeslots):
                policy.check_memory_restart(sess,sim)       
                policy.update_handler(sess,sim)
                # save an instance per training episode for testing purposes.
                if(sim %args.episode_timeslots == 0):
                    model_destination = ('%s/%s_episode%d.ckpt'%(
                            folder_name,json_file_policy,int(float(sim)/args.episode_timeslots))).replace('[','').replace(']','')
                    policy.save(sess,model_destination)
                    
                    # Determine neighbors by using large scale fading.
                    env.reset()
                    env_util = wireless_env.utils2.state_and_reward(env, reward_mode=options_policy['reward_mode'])
                    
                if(sim %args.episode_timeslots < 1):
                    
                    p_strategy,p_strategy_current,SINR_strategy,SINR_strategy_current = initialize_p_sinr()
                    
                if (sim %args.episode_timeslots > 1):                    
                    # Each agent picks its strategy.
                    state = env_util.get_state()
                    reward = env_util.get_reward()
                    for agent in range (env.N):
                        current_local_state = state[agent,0,:]
                        a_time = time.time()  
                        strategy = policy.act(sess,current_local_state,sim,agent)
                        time_calculating_strategy_takes.append(time.time()-a_time)
                        
                        if (sim %args.episode_timeslots > 2): 
                            
                            current_reward = reward[agent] 
                            #no need to apply penalty
                            # if sim > 50:
                            #     current_reward -= penalties[agent]
                            # if sum(p_strategy_all[-1]) > env.N * env.Pmax * 0.95 or sum(p_strategy_all[-1]) < env.N * env.Pmax * 0.01:
                            #     current_reward = -10
                            if not (current_reward == 0 and old_reward[agent] == 0):
                                policy.remember(agent,current_local_state,current_reward)
                            
                        # Only train it once per timeslot, but randomly choose reward from an agent and put reward in training
                        np.random.seed(sim)
                        rd = np.random.randint(0,env.N-1,1)
                        if agent == (rd[0]): 
                            a_time = time.time()
                            # train for a minibatch
                            policy.train(sess,sim)                           
                            time_optimization_at_each_slot_takes.append(time.time()-a_time)                              
                        # Pick the action
                        p_strategy[agent] = policy.strategy_translation[strategy][0] #** 10
                        SINR_strategy[agent] = policy.strategy_translation[strategy][1]
                        # Add current state to the short term memory to observe it during the next state
                        policy.previous_state[agent,:] = current_local_state
                        policy.previous_action[agent] = strategy
                
                #penalty is not applied
                #penalties = np.zeros(env.N)
                for n in range(env.N):
                    # if len(env.packets[n]) == 0 and p_strategy[n] > 0:
                    #     penalties[n] += 1000#p_strategy[n]
                    #penalties[n] = 0
                    # if len(env.packets[n]) > sum([len(env.packets[neigh]) for neigh in env.link_neighbors[n]]) and p_strategy[n] != 0.95 * env.Pmax:
                    #     # penalties[n] += 10
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
                        # penalties[env.user_mapping[k]] += sum(p_strategy[env.user_mapping[k]]) - env.Pmax
                        p_strategy[env.user_mapping[k]] = env.Pmax * p_strategy[env.user_mapping[k]] / sum(p_strategy[env.user_mapping[k]])
                    # if sum(p_strategy[env.user_mapping[k]]) + sum(p_strategy[env.neighbors[k]]) < 0.01:
                    #     print('penalty')
                    #     penalties[env.user_mapping[k]] = 10.0
    
                        
                p_strategy_current = np.array(p_strategy)
                
                #store the output
                sum_rate_distributed_policy.append(np.array(env.throughput))
                p_strategy_all.append(np.array(p_strategy))
                SINR_strategy_all.append(np.array(SINR_strategy))
                
                if sum(p_strategy) > env.K * env.Pmax * 0.95:
                    print('all 1 sim %d'%(sim))
                if sum(p_strategy) < env.K * env.Pmax * 0.01 and sum([len(arr) for arr in env.packets])!=0:
                    print('all 0 sim %d'%(sim))
                if sum([len(arr) for arr in env.packets])==0:
                    print('clean queues sim %d'%(sim))
                    
                old_reward = np.array(env_util.get_reward())
                
                env.step(p_strategy.reshape(env.N,env.M),SINR_strategy.reshape(env.N,env.M))
                # feedbacks = env.feedbacks #number of countinuous failed transmissions
                if(sim % args.episode_timeslots == 0):
                    print('Timeslot %d'%(sim))
                    
            policy.equalize(sess)
            print('Train is over')
        
            model_destination = ('%s/%s_episode%d.ckpt'%(
                    folder_name,json_file_policy,int(float(sim+1)/args.episode_timeslots))).replace('[','').replace(']','')
            policy.save(sess,model_destination)
               
        # End Train Phase
        np_save_path = '%s/%s.npz'%(folder_name,json_file_policy)
        print(np_save_path)
        np.savez(np_save_path,options_policy,sum_rate_distributed_policy,p_strategy_all,SINR_strategy_all,
                 time_optimization_at_each_slot_takes,time_calculating_strategy_takes)
        

    