# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 09:06:33 2022

@author: zyimi
"""
# -*- coding: utf-8 -*-
"""
@author: sinannasir
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import json
import matplotlib
# matplotlib.use('Qt5Agg')
import argparse

parser = argparse.ArgumentParser(description='give test scenarios.')
parser.add_argument("--seeds", default=[1], nargs='+', type=int)              # Sets Gym, PyTorch and Numpy seeds
# parser.add_argument("--max-rates", default=[5, 10, 15, 20, 25, 30 , 35, 40 , 45, 50, 60], nargs='+', type=float)              # Sets Gym, PyTorch and Numpy seeds
parser.add_argument("--max-rates", default=[ 2, 5, 10, 15, 20, 25, 30, 35], nargs='+', type=float)              # Sets Gym, PyTorch and Numpy seeds
# parser.add_argument("--max-rates", default=[1], nargs='+', type=float)              # Sets Gym, PyTorch and Numpy seeds

parser.add_argument("--mode", default="traffic")
parser.add_argument("--episode-timeslots", default=2500, type=int)   # Max time steps to run environment
parser.add_argument("--timeslots", default=100000, type=int)   # Max time steps to run environment
parser.add_argument("--reset-gains", default=True, type=bool) 
parser.add_argument("--K", default=5, type=int)   # Max time steps to run environment
parser.add_argument("--N", default=10, type=int)   # Max time steps to run environment
parser.add_argument("--M", default=1, type=int)   # Max time steps to run environment
parser.add_argument("--pre_sinr_f", default=True,  type=bool)   # Max time steps to run environment




# parser.add_argument("--policy-seeds", default=[1994], nargs='+', type=int)              # Sets Gym, PyTorch and Numpy seeds
# parser.add_argument("--policy-episodes", default=[13], nargs='+', type=int)              # Sets Gym, PyTorch and Numpy seeds
parser.add_argument("--policy-seeds", default=[1], nargs='+', type=int)              # Sets Gym, PyTorch and Numpy seeds
parser.add_argument("--policy-episodes", default=[0, 1, 10, 15, 20, 25, 30, 35, 40], nargs='+', type=int)              # Sets Gym, PyTorch and Numpy seeds
parser.add_argument("--policy-episode-timeslots", default=2500, type=int)   # Max time steps to run environment
parser.add_argument("--policy-timeslots", default=100000, type=int)   # Max time steps to run environment
parser.add_argument("--policy-mode", default="traffic")
# parser.add_argument("--policy-max-rate", default=40, type=int) 
parser.add_argument("--policy-max-rate", default=[5, 10, 15, 20, 25, 30, 35], type=int) 

parser.add_argument("--policy-reset-gains", default=True, type=bool) 
parser.add_argument("--policy-K", default=5, type=int)   # Max time steps to run environment
parser.add_argument("--policy-N", default=10, type=int)   # Max time steps to run environment
parser.add_argument("--policy-M", default=1, type=int)   # Max time steps to run environment
# parser.add_argument('--json-file-policy', type=str, default='dqn200_100_50_rewardtotal_Mbytes_batch128_lr001',
#                     help='json file for the hyperparameters')
parser.add_argument('--json-file-policy', type=str, default='dqn_total_Mbits_b128_lr001_e8',
                    help='json file for the hyperparameters')


args = parser.parse_args()
    
# def main(scenario):    
json_file_policy = args.json_file_policy
num_sim = len(args.seeds)

total_samples = args.timeslots

history = 250
eps = args.policy_episodes

mean_t_sumrate = np.zeros(len(args.max_rates))

mean_t_policy = np.zeros((len(eps),len(args.max_rates)))



mean_delay_policy = np.zeros((len(eps),len(args.max_rates))) 
instable_policy = np.zeros(len(eps)).astype(int)





t_start = args.timeslots // 2


num_simulations = len(args.seeds)

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

for idx, max_rate in enumerate(args.max_rates):
    args.max_rate = max_rate
    for seed in args.seeds:
        args.seed = seed
        
        for idxep, ep in enumerate(eps):
            for policy_seed in args.policy_seeds:
                args.policy_seed = policy_seed
                folder_name_traffic = './simulations/N%dK%dM%dseed%dtimeslots%depisode_timeslots%dmode%sreset_gains%smax_rate%dpre_sinr_f%s'%(args.N,args.K,args.M,seed,args.timeslots,args.episode_timeslots,'traffic',args.reset_gains,max_rate,args.pre_sinr_f)
                policy_folder_name = './simulations/N%dK%dM%dseed%dtimeslots%depisode_timeslots%dmode%sreset_gains%smax_rate%dpre_sinr_f%s'%(args.policy_N,args.policy_K,args.policy_M,args.policy_seed,args.policy_timeslots,args.policy_episode_timeslots,args.policy_mode,args.policy_reset_gains,max_rate,args.pre_sinr_f)
                # np_save_path = '%s/%s%s.npz'%(folder_name_traffic,policy_folder_name.split('./simulations/')[1],json_file_policy)
                np_save_path = '%s/%s%de%dr%d.npz'%(folder_name_traffic,json_file_policy,policy_seed,ep,max_rate)
                data = np.load(np_save_path)
    
                throughput_policy = data['arr_1'][t_start:,:]
                processed_packets_t_policy = []
                waiting_packets = []
                
                p = data['arr_2']
                xyz = data['arr_3']
                # print(len(xyz[0]))
                TX_locs =               data['arr_5']
                RX_locs =               data['arr_6']
                total_packets =         data['arr_7']
                total_Mbits =           data['arr_8']
                average_wait_time =     data['arr_9']
                total_served =          data['arr_10']
                average_delay =         data['arr_11']
                direct_channel_gains=   data['arr_12'] ** 2
                spec_effs =             data['arr_13']
                interfs =               data['arr_14']
#                link_is_activated =     data['arr_15']
                
                # x0 = data['arr_3'][0]
                # x1 = data['arr_3'][1]
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
                env.reset()
                

                # d = list(set(total_packets.flatten()))
                #     # if len(d) < 10:
                #     #     processed_packets_t_policy.append(args.episode_timeslots * max_rate)
                # for dd in d:
                #     waiting_packets.append(dd)
                #         # not stable
                # if max(waiting_packets) > 100 and instable_policy[idxep]==0:
                #         instable_policy[idxep] = idx + 1
                #         print("from ep{} max_rate{} become not stable".format(ep, max_rate))

                
                for d in data['arr_3']:
                    # if len(d) < 10:
                    #     processed_packets_t_policy.append(args.episode_timeslots * max_rate)
                    for dd in d:
                        processed_packets_t_policy.append(dd)
                        


                #not stable
                if max(processed_packets_t_policy) > 1000 and instable_policy[idxep]==0:
                    instable_policy[idxep] = idx + 1
                    print("from ep{} max_rate{} become not stable".format(ep, max_rate))
                    
                
                        
                mean_t_policy[idxep,idx] = mean_t_policy[idxep,idx] + np.mean(throughput_policy) / (len(args.policy_seeds) * num_simulations)
                mean_delay_policy[idxep,idx] = mean_delay_policy[idxep,idx] + np.mean(processed_packets_t_policy) / (len(args.policy_seeds) * num_simulations) 
                

    

lines = ["-"]#,"--",':','-.',':','-.']
linecycler = cycle(lines)
fig = plt.figure()

# plt.plot(np.array(args.max_rates)/2, mean_delay_sumrate*0.02, label='sumrate maximization',linestyle=next(linecycler), marker='x',linewidth=2, markersize=6)
for idxep, ep in enumerate(eps):
    if instable_policy[idxep] != 0:
        mean_delay_policy[idxep][instable_policy[idxep]-1] = 2* mean_delay_policy[idxep][instable_policy[idxep]-1]
        plt.plot(np.array(args.max_rates[:instable_policy[idxep]]), mean_delay_policy[idxep][:instable_policy[idxep]]*0.02, label='policy after episode %d'%(ep),linestyle=next(linecycler), marker='o',linewidth=2, markersize=6)
    else:
        plt.plot(np.array(args.max_rates), mean_delay_policy[idxep]*0.02, label='policy after episode %d'%(ep),linestyle=next(linecycler), marker='x',linewidth=2, markersize=6)



plt.xlabel('average traffic load per link (packets/second)')
plt.ylabel('average packet delay (seconds)')
plt.grid(True)
plt.legend(loc=2)
plt.xlim((0.0, 35))
plt.ylim((0.0, 10))
# plt.yticks([0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6])
plt.tight_layout()
plt.savefig('./fig/withfeedback_delay_rates%s_seeds%s'%(args.max_rates,args.seeds)+'.pdf', format='pdf', dpi=1000)
plt.savefig('./fig/withfeedback_delay_rates%s_seeds%s'%(args.max_rates,args.seeds)+'.png', format='png', dpi=1000)
plt.show()


# print('Average WMMSE iterations per run: %.2f'%(np.mean(mean_iterations_WMMSE)))
# print('Average WMMSE iterations per run: %.2f'%(np.mean(mean_iterations_WMMSE)))
    
# if __name__ == "__main__": 
    
    # parser = argparse.ArgumentParser(description='give test scenarios.')
    # parser.add_argument('--json-file', type=str, default='train_K5_N10_shadow10_episode1-10000_travel0_fd10_equal',
    #                    help='json file for the deployment the policies are tested on')
    # parser.add_argument('--json-file-policy', type=str, default='ddpg200_100_50',
    #                    help='json file for the hyperparameters')
    # parser.add_argument('--json-file-policy2', type=str, default='ddpg200_100_50_aggv1_rewardglob',
    #                    help='json file for the hyperparameters')
    # parser.add_argument('--num-sim', type=int, default=6,
    #                    help='If set to -1, it uses num_simulations of the json file. If set to positive, it runs one simulation with the given id.')
    
    # args = parser.parse_args()
    
    # test_scenario = {'json_file':args.json_file,
    #                  'json_file_policy':args.json_file_policy,
    #                  'json_file_policy2':args.json_file_policy2,
    #                  'num_sim':args.num_sim}
    # main(test_scenario)
