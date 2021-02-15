import socket
import tqdm
import threading
import os
import pickle
import secrets
import sched
import queue
import torchvision
import torch

import select

import model as m
import numpy as np

from time import time
from time import sleep

from torchvision import datasets, transforms
from torch import nn, optim
from os import path
import traceback
import re

from flask import Flask, request, send_file, make_response
import auth
import fl_tools
import yaml

import sys

sys.path.append("../")
import values
from values import BUFFER_SIZE, NO_MODEL_UPDATE_AVAILABLE, TOKEN_BUFFER_SIZE, SEPARATOR
from values import NEW_CLIENT, LOCAL_MODEL_RECEIVED, GLOBAL_MODEL_SENT



## CLIENT STATUS
# NEW_CLIENT = 0
# GLOBAL_MODEL_SENT = 1
# LOCAL_MODEL_RECEIVED = 2

# TODO
# WRTIE YAML STUFF, get host and port as well
# SETUP SERVER WITH YAML
# DEEP LEARNING STUFF

# TOKEN_BUFFER_SIZE = 16
# BUFFER_SIZE = 4096
# SEPARATOR = "&"

app = Flask(__name__)

@app.route('/register_client', methods = ['GET'])
def register_client(self):
    registration_token = request.headers['Token']
    if registration_token == values.REQUIRES_TOKEN: # REGISTER CLIENT
        token = auth.gen_token(server.tokens)
        if token != False:
            server.add_token_to_list(token)
            return str(token)
    return values.invalid_request

@app.route('/get_model', methods = ['GET'])
def send_model():
    token = request.headers['Token']
    valid = auth.check_token_validity(token, server.get_client_data())
    if valid: # SEND MODEL

        filepath = server.model_path
        filename = os.path.basename(filepath)
        filesize = os.path.getsize(filepath)

        response = make_response(send_file(filepath))
        response.headers['Token'] = values.receive_token_valid
        response.header['Filesize'] = filesize

    else:
        response = make_response()
        response.status_code = values.INVALID_CLIENT
        response.headers['Token'] = values.receive_token_invalid
    
    return response

@app.route('/send_model', methods = ['POST'])
def recieve_model():
    token = request.headers['Token']
    file_size = request.headers['Filesize']
    response = make_response()
    valid = auth.check_token_validity(token, server.get_client_data())
    if valid: # RECEIVE MODEL
        model = request.files['model']
        path_to_save = os.path.join(server.client_updates_path, token + ".pth")
        model.save(path_to_save)
        saved_size = os.path.getsize(path)
        if saved_size == file_size: # FILE OK
            server.server_receive_ok(token)
            response.status_code = values.OK_STATUS
        else:
            response.status_code = values.ERROR_STATUS
            
    else: # INVALID CLIENT
        response.status_code = values.INVALID_CLIENT
        response.headers['Token'] = values.client_invalid_token

    return response

@app.route('/check_update', methods = ['POST'])
def check_update():
    token = request.headers['Token']
    valid = auth.check_token_validity(token, server.get_client_data())
    if valid: # SEND UPDATE STATUS
        if server.global_update:
            return values.MODEL_UPDATE_AVAILABLE
        else:
            return values.NO_MODEL_UPDATE_AVAILABLE
    else: # INVALID CLIENT
        response = make_response()
        response.status_code = values.INVALID_CLIENT


@app.route('/model_received', methods = ['GET'])
def client_received_model():
    token = request.headers['Token']
    valid = auth.check_token_validity(token, server.get_client_data())
    response = make_response()
    if valid: # CHANGE CLIENT DATA
        server.client_receive_ok(token)
        response.headers['Token'] = values.receive_token_valid

    else:
        response.status_code = values.INVALID_CLIENT
        response.headers['Token'] = values.receive_token_invalid
    
    return response

class Server:
    def __init__(self, config):
        super().__init__()
        self.config = config
        # CREATE LOCK OBJECT
        self.lock = threading.Lock()
        self.global_update = False

        # OPEN TOKEN DATA
        if os.path.isfile("tokens.pkl"):  # TOKENS FILE EXISTS
            with open("tokens.pkl", "rb") as file:
                self.tokens = pickle.load(file)
        else:
            self.tokens = {}  # CREATE EMPTY TOKENS LIST

        # OPEN CLIENT DATA
        if os.path.isfile("client_data.pkl"):  # CLIENT DATA EXISTS
            with open("client_data.pkl", "rb") as file:
                self.client_data = pickle.load(file)
        else:
            self.client_data = {}  # STORE CLIENT INFO

        print("\nINITIALISING SERVER...")

        self.HOST = self.config["HOST"]
        self.PORT = self.config["PORT"]

        # CLIENT UPDATES FOLDER PATH
        try:
            self.client_updates_path = self.config["client_updates_path"]
            print("\nCLIENT UPDATES FOLDER FOUND:", self.config["client_updates_path"])
            if not os.path.isdir(self.client_updates_path):
                print("\nPATH NOT FOUND, CREATING FOLDER")
                try:
                    os.makedirs(self.client_updates_path)
                except:
                    print("\nUNABLE TO CREATE PATH")
                    if not os.path.isdir("CLIENT_UPDATES"):
                        print("CREATING DEFAULT FOLDER..")
                        os.mkdir("CLIENT_UPDATES")
                    else:
                        print("\nUSING DEFAULT FOLDER")
                    self.client_updates_path = "CLIENT_UPDATES"

        except Exception as e:
            print(e)
            # print('\nFOLDER TO STORE CLIENT UPDATES WAS NOT FOUND')
            if not os.path.isdir("CLIENT_UPDATES"):
                print("CREATING DEFAULT FOLDER..")
                os.mkdir("CLIENT_UPDATES")
            else:
                print("\nUSING DEFAULT FOLDER")
            self.client_updates_path = "CLIENT_UPDATES"

        # MODEL PATH
        if not os.path.isfile(self.config["model_path"]):
            self.model_path = self.config["model_path"]
            print("\nERROR: UNABLE TO FIND MODEL, SERVER WILL CREATE MODEL")

            model = m.Model(self.config["arch"], self.config["n_classes"])
            torch.save(model.state_dict(), self.model_path)
        else:
            self.model_path = self.config["model_path"]

        # TODO:
        # CONFIG FILE FOR DEEP LEARNING
        self.fl_config = ""

        self.aggregation = self.config['aggregation']
        # NO OF ROUNDS OF FL
        self.rounds = self.config['rounds']

        # self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("\nSERVER INITIALISED")

    def save_tokens(self):
        with open("tokens.pkl", "wb") as file:
            pickle.dump(self.tokens, file)

    def save_client_data(self):
        with open("client_data.pkl", "wb") as file:
            pickle.dump(self.client_data, file)

    def get_client_data(self):
        return self.client_data

    def add_token_to_list(self, token):
        self.tokens[token] = token
        self.save_tokens()

    def server_receive_ok(self, token):
        self.lock.acquire()  # BLOCKS UNTIL LOCK IS ACQUIRED
        self.client_data[token] = LOCAL_MODEL_RECEIVED
        self.save_client_data()
        self.lock.release()

        self.lock.acquire()  # BLOCKS UNTIL LOCK IS ACQUIRED
        result = fl_tools.check_client_data(self.get_client_data(),self.aggregation, self.client_updates_path, self.model_path)  # RUN WITH BLOCKING
        self.lock.release()

        if result == True:
            self.rounds -= 1
            self.global_update = True
        print('\nFL ROUNDS LEFT: ', (self.rounds)) # TEMP
        if self.rounds < 0:
            print('FL COMPLETED')


    def client_receive_ok(self, token):
        self.lock.acquire()  # BLOCKS UNTIL LOCK IS ACQUIRED
        self.client_data[token] = GLOBAL_MODEL_SENT
        self.save_client_data()
        self.check_global_update()
        self.lock.release()

        if self.global_update:
            self.check_global_update()

    def check_global_update(self):
        count = 0
        for k, v in self.client_data.items():
            if v == GLOBAL_MODEL_SENT:
                count += 1
        if count == len(self.client_data.keys()):
            self.global_update = False



if __name__ == '__main__':
    with open('server_config.yaml') as file:
        config = yaml.safe_load(file)
    server = Server(config)
    app.run(server.HOST, port = server.PORT, threaded = True)