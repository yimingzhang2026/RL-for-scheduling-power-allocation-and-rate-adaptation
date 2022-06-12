# -*- coding: utf-8 -*-
"""
Created on Thu Jan 1 10:39:33 2021

@author: sinan
"""

import numpy as np
global treshold_sinr
treshold_sinr = 10.0**(30.0/10.0)

def get_random_rayleigh_variable(N,
                                 random_state,
                                 M=1, 
                                 K=None, 
                                 rayleigh_var=1.0):
    if K is None:
        return np.sqrt(2.0/np.pi) * (rayleigh_var * random_state.randn(N, N, M) +
                                                1j * rayleigh_var * random_state.randn(N, N, M))
    else:
        return np.sqrt(2.0/np.pi) * (rayleigh_var * random_state.randn(N, K, M) +
                                                1j * rayleigh_var * random_state.randn(N, K, M))
    
def get_markov_rayleigh_variable(state,
                                 correlation,
                                 N,
                                 random_state,
                                 M=1, 
                                 K=None,
                                 rayleigh_var=1.0):
    if K is None:
        return correlation*state +np.sqrt(1-np.square(correlation)) * np.sqrt(2.0/np.pi) * (rayleigh_var * random_state.randn(N, N, M) +
                                                1j * rayleigh_var * random_state.randn(N, N, M))
    else:
        return correlation*state +np.sqrt(1-np.square(correlation)) * np.sqrt(2.0/np.pi) * (rayleigh_var * random_state.randn(N, K, M) +
                                                1j * rayleigh_var * random_state.randn(N, K, M))
    

# Ray tracing
def _inside_hexagon(x,y,TX_xhex,TX_yhex):
    n = len(TX_xhex)-1
    inside = False
    p1x,p1y = TX_xhex[0],TX_yhex[0]
    for i in range(n+1):
        p2x,p2y = TX_xhex[i % n],TX_yhex[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside

def _get_distance(K,N,TX_loc, RX_loc):
    distance_vector = np.zeros((N,K))
    # tmp_TX_loc = np.zeros((2,N))

        
    # tmp_TX_loc = TX_loc[:,cell_mapping]
    
    for k in range (K):
        distance_vector[:,k]=np.sqrt(np.square(TX_loc[0,k]-RX_loc[0,:])+
                       np.square(TX_loc[1,k]-RX_loc[1,:]))   
            
    return distance_vector


def generate_Cellular_CSI(N, 
                          K,
                          random_state,
                          M = 1, 
                          rayleigh_var = 1.0, 
                          shadowing_dev = 8.0,
                          R = 200, 
                          min_dist = 35,
                          equal_number_for_BS = True):

    assert not equal_number_for_BS or N % K == 0, 'N needs to be divisible by UE_perBS!'

            
    # IMAC Case: we have the mirror BS at the same location.
    max_dist = R
    x_hexagon=R*np.array([0, -np.sqrt(3)/2, -np.sqrt(3)/2, 0, np.sqrt(3)/2, np.sqrt(3)/2, 0])
    y_hexagon=R*np.array([-1, -0.5, 0.5, 1, 0.5, -0.5, -1])

    TX_loc = np.zeros((2,K))
    TX_xhex = np.zeros((7,K))
    TX_yhex = np.zeros((7,K))
    
    RX_loc = np.zeros((2,N))
    cell_mapping = np.zeros(N).astype(int)
    user_mapping = [[] for idx in range(K)]
    
    ############### DROP ALL txers    
    generated_hexagons = 0
    i = 0

    TX_loc [0, generated_hexagons] = 0.0
    TX_loc [1, generated_hexagons] = 0.0
    TX_xhex [:,generated_hexagons] = x_hexagon
    TX_yhex [:,generated_hexagons] = y_hexagon
    generated_hexagons += 1

    while(generated_hexagons < K):
        for j in range(6):
            tmp_xloc = TX_loc [0, i]+np.sqrt(3)*R*np.cos(j*np.pi/(3))
            tmp_yloc = TX_loc [1, i]+np.sqrt(3)*R*np.sin(j*np.pi/(3))
            tmp_xhex = tmp_xloc+x_hexagon
            tmp_yhex = tmp_yloc+y_hexagon
            was_before = False
            for inner_loop in range(generated_hexagons):
                if (abs(tmp_xloc-TX_loc [0, inner_loop*1])<R*1e-2 and abs(tmp_yloc-TX_loc [1, inner_loop*1])<R*1e-2):
                    was_before = True
                    break
            if (not was_before):
                TX_loc [0, generated_hexagons] = tmp_xloc
                TX_loc [1, generated_hexagons] = tmp_yloc
                TX_xhex [:,generated_hexagons] = tmp_xhex
                TX_yhex [:,generated_hexagons] = tmp_yhex      
                generated_hexagons += 1
            if(generated_hexagons>= K):
                break
        i += 1
        
    ############### DROP USERS
    for i in range(N):
        # Randomly assign initial cell placement
        if equal_number_for_BS:
            UE_perBS = N//K
            cell_mapping[i] = int(i/UE_perBS)
        else:
            cell_mapping[i] = random_state.randint(K)
        this_cell = cell_mapping[i]
        user_mapping[this_cell].append(i)
        
        # Place UE within that cell.
        constraint_minx_UE=min(TX_xhex[:,this_cell])
        constraint_maxx_UE=max(TX_xhex[:,this_cell])
        constraint_miny_UE=min(TX_yhex[:,this_cell])
        constraint_maxy_UE=max(TX_yhex[:,this_cell])
        inside_checker = True
        while (inside_checker):
            RX_loc[0,i]=random_state.uniform(constraint_minx_UE,constraint_maxx_UE)
            RX_loc[1,i]=random_state.uniform(constraint_miny_UE,constraint_maxy_UE)
            tmp_distance2center = np.sqrt(np.square(RX_loc[0,i]-TX_loc [0, this_cell])+np.square(RX_loc[1,i]-TX_loc [1, this_cell]))
            if(_inside_hexagon(RX_loc[0,i],RX_loc[1,i],TX_xhex[:,this_cell],TX_yhex[:,this_cell])
                and tmp_distance2center>min_dist and tmp_distance2center<max_dist):
                inside_checker = False

    distance_vector = _get_distance(K,N,TX_loc, RX_loc)
    
    # Get 2D distance pathloss, original pathloss tried in the previous versions
    # Get channel gains
    g_dB2_cell2user = - (128.1 + 37.6* np.log10(0.001*distance_vector))
    shadowing_vector = random_state.randn(N,K)
    
    # rayleigh_vector = np.zeros((N,K,M))
    # rayleigh_rand_var = abs(get_random_rayleigh_variable(N, M))
    
    g_dB2_cell2user = g_dB2_cell2user + shadowing_dev * shadowing_vector
    # Repeat the small scale fading for M subbands
    g_dB2_cell2user = np.repeat(np.expand_dims(g_dB2_cell2user,axis=2),M,axis=-1)
    
    g_dB2 = np.zeros((N,N,M))
    for n in range(N):
        g_dB2[n,:,:] = g_dB2_cell2user[n,cell_mapping,:]
        # rayleigh_vector[n,:,:] = rayleigh_rand_var[cell_mapping[sample_idx],n,:]
        
    gains = np.power(10.0, g_dB2 / 10.0)
    gains_cell2user = np.power(10.0, g_dB2_cell2user / 10.0)
    
    # H_all[sample_idx] = np.multiply(np.sqrt(np.repeat(np.expand_dims(gains,axis=2),M,axis=-1)),rayleigh_vector)


    return gains, gains_cell2user, cell_mapping, user_mapping, TX_loc, RX_loc


# Calculate sum_rate with given channel and power allocation
def sumrate_multi_list_clipped(H,p,noise_var):
    # Take pow 2 of abs_H, no need to take it again and again
    H_2 = H ** 2
    N = H.shape[1] # number of links
    M = H.shape[2] # number of channels
    
    sum_rate1 = np.zeros((N, M)) #calculate SINR
    sum_rate2 = np.zeros((N, M)) #calculate spec_eff
    total_interf = np.zeros((N, M))
    for out_loop in range(M):
        for loop in range (N):
            tmp_1 = H_2[loop, loop, out_loop] * p[loop, out_loop]
            tmp_2 = np.matmul(H_2[loop, :, out_loop], p[:, out_loop]) + noise_var - tmp_1
            total_interf[loop,out_loop] = tmp_2
            if tmp_1 == 0:
                sum_rate1[loop,out_loop] = 0.0
                sum_rate2[loop,out_loop] = 0.0
            else:
                sum_rate1[loop,out_loop] = 10*np.log10(tmp_1/tmp_2)
                sum_rate2[loop,out_loop] = np.log2(1.0+tmp_1/tmp_2)
    return sum_rate1, sum_rate2, total_interf

def sumrate_multi_weighted_clipped(H,p,var_noise,weight):
    return sum(np.multiply(weight, np.sum(sumrate_multi_list_clipped(H,p,var_noise)[0],axis=1)))




    
    
    
    
    
    
    