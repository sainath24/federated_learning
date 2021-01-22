"""
defines the aggregator functions.
"""

import torch 

import torch.nn as nn
import numpy as np

def average(model_paths): 
    """
    :param: model_paths: list of paths to client models
    :desc: returns the mean of all models
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_models = len(model_paths)
    state_dict = torch.load(model_paths[0], map_location=device)
    for i in range(1, len(model_paths)):
        state_dict_2 = torch.load(model_paths[i], map_location=device)

        for key in state_dict:
            state_dict[key] += state_dict_2[key]
            
    for key in state_dict:
        state_dict[key] = state_dict[key] / torch.tensor(total_models,dtype=torch.long)
    return state_dict