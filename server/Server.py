import socket
import tqdm
import threading
import os
import yaml
import pickle
import secrets
import sched
import queue
import torchvision
import torch

import model as m
import numpy as np

from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from os import path

# QUERIES
# MODEL - get latest model
# CONFIG - get latest config
# UPDATE_MODEL - send updated model

# CLIENT DATA
# 1 - MODEL SENT
# 2 - MODEL RECEIVED
MODEL_SENT = 1
MODEL_RECEIVED = 2

#TODO
# WRTIE YAML STUFF, get host and port as well
# SETUP SERVER WITH YAML
# DEEP LEARNING STUFF

BUFFER_SIZE = 4096
SEPARATOR = '&'

class Server:
    def __init__(self):
        super().__init__()

        #CREATE LOCK OBJECT
        self.lock = threading.Lock()

        self.global_update = False

        #SET UP QUEUE
        # self.queue = queue.Queue()


        # OPEN TOKEN DATA
        if os.path.isfile('tokens.pkl'): # TOKENS FILE EXISTS
            with open('tokens.pkl', 'rb') as file:
                self.tokens = pickle.load(file)
            
        else:
            self.tokens = {} # CREATE EMPTY TOKENS LIST

        # OPEN CLIENT DATA
        if os.path.isfile('client_data.pkl'): # CLIENT DATA EXISTS
            with open('client_data.pkl', 'rb') as file:
                self.client_data = pickle.load(file)
        else:
            self.client_data = {} # STORE CLIENT INFO

        print('\nINITIALISING SERVER...')
        with open('server_config.yaml') as file:
            config = yaml.safe_load(file)

        self.HOST = config["HOST"]
        self.PORT = config["PORT"]
        
        # CLIENT UPDATES FOLDER PATH
        try:
            self.client_updates_path = config['client_updates_path']
            print('\nCLIENT UPDATES FOLDER FOUND:', config['client_updates_path'])
            if not os.path.isdir(self.client_updates_path):
                print('\nPATH NOT FOUND, CREATING FOLDER')
                try:
                    os.makedirs(self.client_updates_path)
                except:
                    print('\nUNABLE TO CREATE PATH')
                    if not os.path.isdir('CLIENT_UPDATES'):
                        print('CREATING DEFAULT FOLDER..')
                        os.mkdir('CLIENT_UPDATES')
                    else:
                        print('\nUSING DEFAULT FOLDER')
                    self.client_updates_path = 'CLIENT_UPDATES'

        except Exception as e:
            print(e)
            # print('\nFOLDER TO STORE CLIENT UPDATES WAS NOT FOUND')
            if not os.path.isdir('CLIENT_UPDATES'):
                print('CREATING DEFAULT FOLDER..')
                os.mkdir('CLIENT_UPDATES')
            else:
                print('\nUSING DEFAULT FOLDER')
            self.client_updates_path = 'CLIENT_UPDATES'
            
        # MODEL PATH
        if not os.path.isfile(config['model_path']):
            self.model_path = config['model_path']
            print('\nERROR: UNABLE TO FIND MODEL, SERVER WILL CREATE MODEL')

            model = m.Model(config["arch"], config["n_classes"])
            torch.save(model.state_dict(), self.model_path)
        else:
            self.model_path = config['model_path']

        #TODO:
        # CONFIG FILE FOR DEEP LEARNING
        self.fl_config = ''

        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print('\nSERVER INITIALISED')

    def save_tokens(self):
        with open('tokens.pkl','wb') as file:
            pickle.dump(self.tokens, file)

    def save_client_data(self):
        with open('client_data.pkl', 'wb') as file:
            pickle.dump(self.client_data, file)
    
    def get_client_data(self):
        return self.client_data

    def send_token(self, conn, addr): # SEND NEW TOKEN TO CLIENT
        try:
            # GENERATE NEW TOKEN
            token = secrets.token_hex(8)
            self.tokens[token] = addr[0]
            self.save_tokens()
            # SEND TOKEN TO CLIENT
            conn.send(token.encode())
            return token
        except Exception as e:
            print('\nEXCEPTION IN send_token: ', e)
        return False

    def check_global_update(self):
        count = 0
        for k,v in self.client_data.items():
            if v == MODEL_SENT:
                count+=1
        if count == len(self.client_data.keys()):
            self.global_update = False



    def send_model(self, conn, addr, token):
        try:
            if not os.path.isfile(self.client_updates_path + '/' + token + '.pth') or self.global_update:
                filepath = self.model_path
                filename = os.path.basename(filepath)
                filesize = os.path.getsize(filepath)
                conn.send(f"{filename}{SEPARATOR}{filesize}".encode())
                print('\nBUFFER SIZE: ', BUFFER_SIZE, ' GLOBAL UPDATE: ', self.global_update)
                progress = tqdm.tqdm(range(filesize), "SENDING MODEL TO " + addr[0])
                with open(filepath, 'rb') as file:
                    while True:
                        read = file.read(BUFFER_SIZE)
                        if not read:
                            break
                        conn.sendall(read)
                        progress.update(len(read))
                    
                # GET LOCK AND WRITE TO CLIENT DATA
                self.lock.acquire() # BLOCKS UNTIL LOCK IS ACQUIRED
                self.client_data[token] = MODEL_SENT
                self.save_client_data()
                self.check_global_update()
                # if self.check_client != None:
                #     self.check_client() # RUN WITH BLOCKING
                self.lock.release()
            elif not self.global_update: # NOT UPDATED
                conn.send(f"NOT_UPDATED".encode())
                print('\nNOT SENDING UPDATE GLOBAL UPDATE: ', self.global_update)
                
            
        except Exception as e:
            print('\nEXCEPTION IN send_model: ', e)
            return False

        return True

    def send_config(self, conn, addr, token):
        try:
            filepath = self.fl_config
            filename = os.path.basename(filepath)
            filesize = os.path.getsize(filepath)
            conn.send(f"{filename}{SEPARATOR}{filesize}".encode())
            progress = tqdm.tqdm(range(filesize), "SENDING MODEL TO " + addr[0])
            with open(filename, 'rb') as file:
                while True:
                    read = file.read(BUFFER_SIZE)
                    if not read:
                        break
                    conn.sendall(read)
                    progress.update(len(read))
        except Exception as e:
            print('\nEXCEPTION IN send_config: ', e)
            return False

        return True
    
    def receive_updated_model(self, conn, addr, token):
        # TODO:
        # SET UPDATES COUNTER TO UPDTAE GRADIENT
        try:
            #RECEIVE DATA ON FILE
            data = conn.recv(BUFFER_SIZE).decode()
            filename, filesize = data.split(SEPARATOR)
            filesize = int(filesize)

            path_to_save = os.path.join(self.client_updates_path, token + '.pth')

            print('\nRECEIVED NEW MODEL INFO: ', filename, ' SIZE: ', filesize)

            #RECEIVE FILE
            progress = tqdm.tqdm(range(filesize), "RECEIVING UPDATED MODEL FROM: " + addr[0])
            with open(path_to_save, 'wb') as file:
                while True:
                    data = conn.recv(BUFFER_SIZE)
                    if not data:
                        break
                    file.write(data)
                    progress.update(len(data))

            # GET LOCK AND WRITE TO CLIENT DATA
            self.lock.acquire() # BLOCKS UNTIL LOCK IS ACQUIRED
            self.client_data[token] = MODEL_RECEIVED
            self.save_client_data()
            # if self.check_client != None:
            self.check_client() # RUN WITH BLOCKING
            self.lock.release()
            

        except Exception as e:
            print('\nEXCEPTION IN receive_update_model: ', e)
            return False
        return True
        

    def handle_client(self, conn, addr):
        # HANDLES CLIENT, OPEN A SEPARATE THREAD FOR EVERY CLIENT
        print('\nCONNECT TO CLIENT: ' + addr[0])
        result = False
        token = conn.recv(BUFFER_SIZE).decode() #GET TOKEN
        # while query!='DISCONNECT':
        print('\nRECEIVED TOKEN: ' + token)
        #HANDLE TOKEN FIRST
        if token == 'TOKEN': # NEW CLIENT, SEND TOKEN
            token = self.send_token(conn, addr)
            if not token:
                print('\nERROR IN SENDING TOKEN')

        else: # received token
            conn.send(f"TOKEN_OK".encode())
            if token not in self.client_data.keys(): # INSERT CLIENT TOKEN IN DICTIONARY
                self.client_data[token] = ''

        # HANDLE MAIN QUERIES
        query = conn.recv(BUFFER_SIZE).decode() #GET QUERY
        if query == 'MODEL':
            result = self.send_model(conn, addr, token)
        elif query == 'CONFIG':
            result = self.send_config(conn, addr, token)
        elif query == 'UPDATE_MODEL':
            result = self.receive_updated_model(conn, addr, token)
        if result:
            print("QUERY SUCCESSFULLY EXECUTED FOR: " + addr[0] + ' TOKEN: ' + token)
        else:
            print("QUERY FAILED FOR: " + addr[0] + ' TOKEN: ' + token)
            # query = conn.recv(BUFFER_SIZE).decode()
        conn.close()
        
    def start(self, host, port, connections,check_client_data = None):
        
        try:
            self.check_client = check_client_data
            self.s.bind((host, port))
            self.s.listen(connections)
            print('\nSERVER STARTED\n')
            while True:
                conn, addr = self.s.accept()
                client_thread = threading.Thread(target = self.handle_client, args = [conn, addr])
                client_thread.start()
        except Exception as e:
            print('\nEXCEPTION IN start: ', e)
            return False
        return True

