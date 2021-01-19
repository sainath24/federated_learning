import Server
import threading

import numpy as np
import torch
import torchvision
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

class fl_server:
    def __init__(self, rounds = 15, max_client = 5) -> None:
        super().__init__()
        self.rounds = rounds
        self.max_clients = max_client
        self.current_client = 0
        self.server = Server.Server()
        # ACCESS CLIENT DATA WITH self.server.client_data

    def global_update(self, models):
        total_models = len(models)
        sum = 0 
        # state_dict = model.state_dict()
        state_dict = torch.load(self.server.client_updates_path + '/' + models[0])
        for i in range(1, len(models)):
            # print('\nPLEASE DONT GO INSIDE')
            # model.load_state_dict(torch.load(self.server.client_updates_path + '/' + models[i]))
            state_dict_2 = torch.load(self.server.client_updates_path + '/' + models[i])
            print('\nADDING STATE DICTS\n')
            for key in state_dict:
                # print('\nSD1\n')
                # print(state_dict[key])
                # print('\nSD2\n')
                # print(state_dict_2[key])
                state_dict[key] += state_dict_2[key]
                # print('\nSUM\n')
                # print(state_dict[key])
        
        for key in state_dict:
            # print('\nAVERAGING STATE DICTS\n')
            # print(state_dict[key])
            state_dict[key] /= total_models
            # print(state_dict[key])

        # SAVE GLOBAL UPDATE
        torch.save(state_dict, self.server.model_path)
        print('\nNEW GLOBAL MODEL MADEEEEE')
            

    def update_central(self, models):
        print('\nGONNA UPDATEEE YAAAAYYYY')
        client_data = self.server.get_client_data()
        # MAKE A LIST OF MODELS USED FOR CENTRAL UPDATE
        # models = []
        # for k,v in client_data.items():
        #     if v==2:
        #         models.append(str(k) + '.pth')
                
        # TODO: UPDATE CENTRAL MODEL using models list
        # UPDATE MODELS
        print('\nLEN OF MODELS: ', models)
        print('\nTOTAL: ', client_data)
        if len(models) == len(client_data.keys()): # DO GLOBAL UPDATE
            self.global_update(models)
            self.server.global_update = True
            self.rounds -=1
        else:
            self.global_update = False
        if self.rounds<0: # FL IS OVER
            print('\nFEDERATED LEARNING OVER')
            exit()

    def check_client_data(self):
        print('\nSERVER IN CHECK CLIENT DATA')
        waiting_clients = 0
        models = []
        try:
            data = self.server.get_client_data()
            total_clients = len(data.keys())
            print(data)
            for k,v in data.items():
                if v == 2:
                    waiting_clients+=1
                    models.append(str(k) + '.pth')
            # TODO: CHECK CLIENTS
            # CHECK IF YOU NEED TO UPDATE CENTRAL MODEL
            print('\nTOTAL CLIENTS: ', total_clients)
            print('\nWAITING CLIENTS: ', waiting_clients)
            if waiting_clients == total_clients : #and total_clients > 1: # EVERYONE HAS SENT MODEL
                self.update_central(models)

        except Exception as e:
            print('check_client_data: ', e)

    def start_server(self):
        try:
            self.server.start(self.server.HOST, self.server.PORT, self.max_clients, self.check_client_data)
        except Exception as e:
            print('\nEXCEPTING in start_server: ' ,e)

    def start(self):
        # START SERVER ON SEPARATE THREAD
        server_thread = threading.Thread(target=self.start_server)
        server_thread.start()