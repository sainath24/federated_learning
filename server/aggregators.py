"""
defines the aggregator functions.
"""

import torch 

import torch.nn as nn
import numpy as np

def (model_paths): 
    state_dict = torch.load(model_paths[0])
    for i in range(1, len(model_paths)):
        state_dict_2 = torch.load(model_paths[i])

        for key in state_dict:
            # print('\nSD1\n')
            # print(state_dict[key])
            # print('\nSD2\n')
            # print(state_dict_2[key])
            state_dict[key] += state_dict_2[key]
            # print('\nSUM\n')
            # print(state_dict[key])
