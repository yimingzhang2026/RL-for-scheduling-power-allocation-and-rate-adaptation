# -*- coding: utf-8 -*-
"""
Created on Sun May  8 13:23:45 2022

@author: zyimi
"""

import tensorflow as tf

from tensorflow.python import pywrap_tensorflow
alloc_path = './simulations/N10K5M1seed1timeslots100000episode_timeslots2500modetrafficreset_gainsTruemax_rate20pre_sinr_fTrue/dqn_total_Mbits_b128_lr001_e8_episode40.ckpt'
model_reader = pywrap_tensorflow.NewCheckpointReader(alloc_path)

var_dict = model_reader.get_variable_to_shape_map()

for key in var_dict:
    print("variable name: ", key)

    print(model_reader.get_tensor(key))