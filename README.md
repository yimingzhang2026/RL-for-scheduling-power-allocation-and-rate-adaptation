# RL-for-schudling-power-allocation-and-rate-adaptation
This is the code for arXiv paper REINFORCEMENT LEARNING FOR SCHEDULING, POWER CONTROL, AND RATE ADAPTATION
## Required Environment
numpy==1.16.2
tensorflow==1.13.2
scipy
matplotlib
PyQt5
## To train a policy from scratch:
Run train.py
## To get the performance of trained model:
Run test.py
## To plot the traffic-delay curve:
Run sim_result_eps.py
## To change the deployment of wireless network and Hyperparameters for learning
Edit file  ./config/policy/dqn_total_Mbits_b128_lr001_e8.json. You can also use other configuration file.
