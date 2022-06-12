# -*- coding: utf-8 -*-
"""
@author: sinannasir
"""
import numpy as np
#import matplotlib.pyplot as plt
import project_backend as pb
import tensorflow as tf
import collections
import copy
import itertools

class DQN:
    def __init__(self, args,options_policy,env):
        tf.reset_default_graph()        
        self.timeslots = args.timeslots
        self.train_episodes = args.episode_timeslots

        self.N = env.N
        self.Pmax = env.Pmax
        self.noise_var = env.noise_var
        self.args = args
        self.env = env
        
        # sing_ac = options_policy['num_actions']
        
        # powers = np.zeros(sing_ac)
        # powers[0] = 0.0 # Tx power 0
        # Pmin_dB = 10.0-30
        # # Calculate steps in dBm
        # if sing_ac > 2:
        #     strategy_translation_dB_step = (env.Pmax_dB-Pmin_dB)/(sing_ac-2)
        # for i in range(1,sing_ac-1):
        #     powers[i] = np.power(10.0,((Pmin_dB+(i-1)*strategy_translation_dB_step))/10)
        # powers[-1] = env.Pmax
        
        # self.strategy_translation = powers
        
        sing_ac1 = options_policy['power_actions']
        sing_ac2 = options_policy['SINR_actions']
        sing_ac = options_policy['num_actions']
        
        powers = np.zeros(sing_ac1)
        powers[0] = 0.0 # Tx power 0
        Pmin_dB = 10.0-30
        # Calculate steps in dBm
        if sing_ac1 > 2:
            strategy_translation_dB_step = (env.Pmax_dB-Pmin_dB)/(sing_ac1-2)
        for i in range(1,sing_ac1-1):
            powers[i] = np.power(10.0,((Pmin_dB+(i-1)*strategy_translation_dB_step))/10)
        powers[-1] = env.Pmax
        
        SINRs = np.zeros(sing_ac2)
        SINRs[0] = -10
        if sing_ac2 > 2:
            trategy_translation_SINR_step = (env.pre_SINRmax_db + abs(SINRs[0]))/(sing_ac2-1)
        for i in range(1,sing_ac2-1):
            SINRs[i] = SINRs[0] + trategy_translation_SINR_step * i
        SINRs[-1] = env.pre_SINRmax_db
        
        def myfunc(list1, list2):
            return [np.append(i,j) for i in list1 for j in list2]
        
        strategies = myfunc(powers,SINRs)
            
            
        self.strategy_translation = strategies
        
        
        self.num_output = self.num_actions = len(self.strategy_translation) # Kumber of actions
        self.discount_factor = options_policy['discount_factor']
        
        self.N_neighbors = env.N_neighbors
        self.num_input = 6 + 4 * self.N_neighbors + 1 #feedbacks
        if env.mode != 'sumrate':
            self.num_input += 1 + self.N_neighbors
        
        learning_rate_0 = options_policy['learning_rate_0']
        learning_rate_decay = options_policy['learning_rate_decay']
        learning_rate_min = options_policy['learning_rate_min']
        self.batch_size = options_policy['batch_size']
        memory_per_agent = options_policy['memory_per_agent']
        # epsilon greedy algorithm
        max_epsilon = options_policy['max_epsilon']
        epsilon_decay = options_policy['epsilon_decay']
        min_epsilon = options_policy['min_epsilon']
        # quasi-static target network update
        self.target_update_count = options_policy['target_update_count']
        self.time_slot_to_pass_weights = options_policy['time_slot_to_pass_weights'] # 50 slots needed to pass the weights
        n_hidden_1 = options_policy['n_hiddens'][0]
        n_hidden_2 = options_policy['n_hiddens'][1]
        n_hidden_3 = options_policy['n_hiddens'][2]

        
        # Experience-replay memory size
        self.memory_len = memory_per_agent*env.N
        # learning rate
        self.learning_rate_all = [learning_rate_0]
        for i in range(1,self.timeslots):
            # if i % self.train_episodes == 0:
            #     self.learning_rate_all.append(learning_rate_0)
            # else:
            self.learning_rate_all.append(max(learning_rate_min,learning_rate_decay*self.learning_rate_all[-1]))
    #            learning_rate_all.append(learning_rate_all[-1])
    
        # epsilon greedy algorithm       
        self.epsilon_all=[max_epsilon]
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
#         for i in range(1,self.timeslots):
#             if i % self.train_episodes == 0:
# #                if int(i/self.train_episodes) == (self.timeslots/self.train_episodes-1):
# #                    self.epsilon_all.append(0.0) # Test scenario
# #                else:
#                 self.epsilon_all.append(max_epsilon)
#             else:
#                 self.epsilon_all.append(max(min_epsilon,epsilon_decay*self.epsilon_all[-1]))
        
        # Experience replay memory
        self.memory = {}
        self.memory['s'] = collections.deque([],self.memory_len+self.N)
        self.memory['s_prime'] = collections.deque([],self.memory_len)
        self.memory['rewards'] = collections.deque([],self.memory_len)
        self.memory['actions'] = collections.deque([],self.memory_len)
        
        self.previous_state = np.zeros((self.N,self.num_input))
        self.previous_action = np.ones(self.N) * self.num_actions
       
        # required for session to know whether dictionary is train or test
        self.is_train = tf.placeholder("bool")   

        self.x_policy = tf.placeholder("float", [None, self.num_input])
        self.y_policy = tf.placeholder("float", [None, 1])
        with tf.name_scope("weights"):
            self.weights_policy = pb.initial_weights (self.num_input, n_hidden_1,
                                               n_hidden_2, n_hidden_3, self.num_output)
        with tf.name_scope("target_weights"): 
            self.weights_target_policy = pb.initial_weights (self.num_input, n_hidden_1,
                                               n_hidden_2, n_hidden_3, self.num_output)
        with tf.name_scope("tmp_weights"): 
            self.weights_tmp_policy = pb.initial_weights (self.num_input, n_hidden_1,
                                               n_hidden_2, n_hidden_3, self.num_output)
        with tf.name_scope("biases"):
            self.biases_policy = pb.initial_biases (n_hidden_1, n_hidden_2, n_hidden_3,
                                          self.num_output)
        with tf.name_scope("target_biases"): 
            self.biases_target_policy = pb.initial_biases (n_hidden_1, n_hidden_2, n_hidden_3,
                                          self.num_output)
        with tf.name_scope("tmp_biases"): 
            self.biases_tmp_policy = pb.initial_biases (n_hidden_1, n_hidden_2, n_hidden_3,
                                          self.num_output)
        # initialize the neural network for each agent
        self.QNN= pb.neural_net(self.x_policy, self.weights_policy, self.biases_policy)
        self.QNN_target = pb.neural_net(self.x_policy, self.weights_target_policy,
                                            self.biases_target_policy)
        self.actions_flatten = tf.placeholder(tf.int32, self.batch_size)
        self.actions_one_hot = tf.one_hot(self.actions_flatten, self.num_actions, 1.0, 0.0)
        self.single_q = tf.reshape(tf.reduce_sum(tf.multiply(self.QNN, self.actions_one_hot), reduction_indices=1),(self.batch_size,1))
        # loss function is simply least squares cost
        self.loss = tf.reduce_sum(tf.square(self.y_policy - self.single_q))
        self.learning_rate = (tf.placeholder('float'))
        # RMSprop algorithm used
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.9,
                                              epsilon=1e-10).minimize(self.loss)

        self.init = tf.global_variables_initializer()
        # quasi-static target update simulation counter = 0
        self.saver = tf.train.Saver()

    def update_epsilon(self,reset):
            if reset:
#                if int(i/self.train_episodes) == (self.timeslots/self.train_episodes-1):
#                    self.epsilon_all.append(0.0) # Test scenario
#                else:
                self.epsilon_all.append(self.max_epsilon)
            else:
                self.epsilon_all.append(max(self.min_epsilon,self.epsilon_decay*self.epsilon_all[-1]))
        
    def initialize_updates(self,sess): # Keed to rund this before calling quasi static.
        self.saver = tf.train.Saver(tf.global_variables())
        self.update_class1 = []
        for (w,tmp_w) in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='weights'),
             tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='tmp_weights')):
            self.update_class1.append(tf.assign(tmp_w,w))
            sess.run(self.update_class1[-1])
        for (b,tmp_b) in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='biases'),
             tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='tmp_biases')):
            self.update_class1.append(tf.assign(tmp_b,b))
            sess.run(self.update_class1[-1])
        self.update_class2 = []
        for (tmp_w,t_w) in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='tmp_weights'),
             tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='target_weights')):
            self.update_class2.append(tf.assign(t_w,tmp_w))
            sess.run(self.update_class2[-1])
        for (tmp_b,t_b) in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='tmp_biases'),
             tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='target_biases')):
            self.update_class2.append(tf.assign(t_b,tmp_b))
            sess.run(self.update_class2[-1])
        self.simulation_target_update_counter = self.target_update_count
        self.process_weight_update = False
        self.simulation_target_pass_counter = self.time_slot_to_pass_weights
        print('first update')
    
    def check_memory_restart(self,sess,sim):   
        if(sim %self.train_episodes == 0 and sim != 0): # Restart experience replay.
            self.memory = {}
            self.memory['s'] = collections.deque([],self.memory_len+self.N)
            self.memory['s_prime'] = collections.deque([],self.memory_len+self.N)
            self.memory['rewards'] = collections.deque([],self.memory_len+self.N)
            self.memory['actions'] = collections.deque([],self.memory_len+self.N)
            
            self.previous_state = np.zeros((self.N,self.num_input))
            self.previous_action = np.ones(self.N) * self.num_actions
    
    def update_handler(self,sess,sim):
        # Quasi-static target Algorithm
        # First check whether target network has to be changed.
        self.simulation_target_update_counter -= 1
        if (self.simulation_target_update_counter == 0):
            for update_instance in self.update_class1:
                sess.run(update_instance)
            self.simulation_target_update_counter = self.target_update_count
            self.process_weight_update = True

        if self.process_weight_update:
            self.simulation_target_pass_counter -= 1
        
        if (self.simulation_target_pass_counter <= 0):
            for update_instance in self.update_class2:
                sess.run(update_instance)
            self.process_weight_update = False
            self.simulation_target_pass_counter = self.time_slot_to_pass_weights
            
    def act(self,sess,current_local_state,sim,agent):
        # Current QNN outputs for all available actions
        current_QNN_outputs = sess.run(self.QNN_target, feed_dict={self.x_policy: current_local_state.reshape(1,self.num_input), self.is_train: False})
        # epsilon greedy algorithm
        if np.random.rand() < self.epsilon_all[-1]:
            strategy = np.random.randint(self.num_actions)
        else:
            strategy = np.argmax(current_QNN_outputs)
        return strategy
    
    def act_noepsilon(self,sess,current_local_state,sim):
        # Current QNN outputs for all available actions
        current_QNN_outputs = sess.run(self.QNN_target, feed_dict={self.x_policy: current_local_state.reshape(1,self.num_input), self.is_train: False})
        return np.argmax(current_QNN_outputs)
    
    def remember(self,agent,current_local_state,current_reward):
        self.memory['s'].append(copy.copy(self.previous_state[agent,:]).reshape(self.num_input))
        self.memory['s_prime'].append(copy.copy(current_local_state))
        self.memory['actions'].append(copy.copy(self.previous_action[agent]))
        self.memory['rewards'].append(copy.copy(current_reward))
    
    def train(self,sess,sim):
        if len(self.memory['s']) >= self.batch_size+self.N:
            # Minus N ensures that experience samples from previous timeslots been used
            idx = np.random.randint(len(self.memory['rewards'])-self.N,size=self.batch_size)
            c_QNN_outputs = sess.run(self.QNN_target, feed_dict={self.x_policy: np.array(self.memory['s_prime'])[idx, :].reshape(self.batch_size,self.num_input),
                                                                 self.is_train: False})
            opt_y = np.array(self.memory['rewards'])[idx].reshape(self.batch_size) + self.discount_factor * np.max(c_QNN_outputs,axis=1)
            actions = np.array(self.memory['actions'])[idx]
            (tmp,tmp_mse) = sess.run([self.optimizer, self.loss], feed_dict={self.learning_rate:self.learning_rate_all[sim],self.actions_flatten:actions,
                                self.x_policy: np.array(self.memory['s'])[idx, :],
                                self.y_policy: opt_y.reshape(self.batch_size,1), self.is_train: True})
    
    def equalize(self,sess):
        for update_instance in self.update_class1:
            sess.run(update_instance)
        for update_instance in self.update_class2:
            sess.run(update_instance)
            
    def save(self,sess,model_destination):
        self.saver = tf.train.Saver(tf.global_variables())
        save_path = self.saver.save(sess, model_destination)
        print("Model saved in path: %s" % save_path)
        
    def load(self,sess,model_destination):
        self.saver = tf.train.Saver(tf.global_variables())
        self.saver.restore(sess, model_destination)
        print('Model loaded from: %s' %(model_destination))
        
 