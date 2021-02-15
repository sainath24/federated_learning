import Server
import threading
import os
import torch
import torchvision
import traceback
import numpy as np
import aggregators as agg

from time import time
from torchvision import datasets, transforms
from torch import nn, optim

import sys
sys.path.append("../")
import values

def global_update(models, aggregation_technique, client_updates_path, server_model_path):
    total_models = len(models)
    sum = 0 
    model_paths = [os.path.join(client_updates_path, model_path)
                            for model_path in models]
    try:
        if aggregation_technique=='mean': 
            averaged_state_dict = agg.average(model_paths)
        else: 
            # default is FedAvg
            averaged_state_dict = agg.average(model_paths)

        # SAVE GLOBAL UPDATE
        torch.save(averaged_state_dict, server_model_path)
        return True
        print('\nNEW GLOBAL MODEL MADE')
    except Exception as e:
        print('\nERROR IN AGGREGATION: ')
        traceback.print_exc()
        return False
        

def check_client_data(client_data, aggregation_technique, client_updates_path, server_model_path):
    print('\nSERVER IN CHECK CLIENT DATA')
    waiting_clients = 0
    models = []
    try:
        total_clients = len(client_data.keys())
        # print(data)
        for k,v in client_data.items():
            if v == values.LOCAL_MODEL_RECEIVED:
                waiting_clients+=1
                models.append(str(k) + '.pth')
        
        # CHECK IF YOU NEED TO UPDATE CENTRAL MODEL
        print('\nTOTAL CLIENTS: ', total_clients)
        print('\nWAITING CLIENTS: ', waiting_clients)
        if waiting_clients == total_clients and len(models) == len(client_data.keys()):# EVERYONE HAS SENT MODEL
            result = global_update(models, aggregation_technique, client_updates_path)
            return result

    except Exception as e:
        print('check_client_data: ', e)
        traceback.print_exc()
    return False
