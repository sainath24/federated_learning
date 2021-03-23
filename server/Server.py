import socket
from types import MethodType
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
import time

import model as m
import numpy as np

from time import time
from time import sleep

from torchvision import datasets, transforms
from torch import nn, optim
from os import path
import traceback
import re
import json

from flask import Flask, request, send_file, make_response
import auth
import fl_tools
import yaml

import sys


sys.path.append("../")
import values
from values import BUFFER_SIZE, NO_MODEL_UPDATE_AVAILABLE, TOKEN_BUFFER_SIZE, SEPARATOR
from values import NEW_CLIENT, LOCAL_MODEL_RECEIVED, GLOBAL_MODEL_SENT

from datetime import timedelta
from flask import make_response, request, current_app
from functools import update_wrapper

from flask_cors import CORS, cross_origin


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
CORS(app, support_credentials=True)


@app.route("/register_client", methods=["GET"])
def register_client():
    registration_token = request.headers["Token"]
    if registration_token == values.REQUIRES_TOKEN:  # REGISTER CLIENT
        token = auth.gen_token(server.tokens)
        if token != False:
            server.add_token_to_list(token)
            return str(token)
    return values.invalid_request


@app.route("/received_token", methods=["GET"])
def client_received_token():
    token = request.headers["Token"]
    valid = auth.check_token_in_list(token, server.get_token_list())
    response = make_response()
    if valid:
        server.add_client_to_list(token)
        response.headers["Token"] = values.receive_token_valid
    else:
        response.headers["Token"] = values.receive_token_invalid
    return response


@app.route("/get_model", methods=["GET"])
def send_model():
    token = request.headers["Token"]
    valid = auth.check_token_validity(token, server.get_client_data())
    if valid:  # SEND MODEL

        data_length = int(request.headers["Datalength"])
        server.set_client_data_length(token, data_length)

        filepath = server.model_path
        filename = os.path.basename(filepath)
        filesize = os.path.getsize(filepath)

        response = make_response(send_file(filepath))
        response.headers["Token"] = values.receive_token_valid
        response.headers["Filesize"] = str(filesize)
        response.status_code = values.OK_STATUS

    else:
        response = make_response()
        response.status_code = values.INVALID_CLIENT
        response.headers["Token"] = values.receive_token_invalid

    return response


def save_model(token, file, path):
    file.save(path)
    server.server_receive_ok(token)


@app.route("/send_model", methods=["POST"])
def receive_model():
    token = request.headers["Token"]
    file_size = request.headers["Filesize"]
    response = make_response()
    valid = auth.check_token_validity(token, server.get_client_data())
    if valid:  # RECEIVE MODEL
        # print('\nVALID CLIENT\n')
        model = request.files["model"]
        print("\nGOT MODEL FROM REQUEST\n")
        path_to_save = os.path.join(server.client_updates_path, token + ".pth")
        print("\nRECEIVED HEADER FILESIZE: ", file_size)
        print("\npath: ", path_to_save)
        # print('\nCALCULATED FILE SIZE: ', len(model.read()))
        # if len(model.read()) == int(file_size): # MODEL IS OK
        # thread = threading.Thread(target = save_model, args=[token, model, path_to_save]).start()
        model.save(path_to_save)
        print("\nMODEL SAVED")
        server.server_receive_ok(token)
        response.status_code = values.OK_STATUS
        # else:
        #     response.status_code = values.ERROR_STATUS

    else:  # INVALID CLIENT
        response.status_code = values.INVALID_CLIENT
        response.headers["Token"] = values.client_invalid_token

    return response


@app.route("/model_received", methods=["GET"])
def client_received_model():
    token = request.headers["Token"]
    valid = auth.check_token_validity(token, server.get_client_data())
    response = make_response()
    if valid:  # CHANGE CLIENT DATA
        server.client_receive_ok(token)
        response.headers["Token"] = values.receive_token_valid

    else:
        response.status_code = values.INVALID_CLIENT
        response.headers["Token"] = values.receive_token_invalid

    return response


@app.route("/check_update", methods=["GET"])
def check_update():
    token = request.headers["Token"]
    valid = auth.check_token_validity(token, server.get_client_data())
    if valid:  # SEND UPDATE STATUS
        if server.check_global_update_for_client(token):
            return values.MODEL_UPDATE_AVAILABLE
        else:
            return values.NO_MODEL_UPDATE_AVAILABLE
    else:  # INVALID CLIENT
        response = make_response()
        response.status_code = values.INVALID_CLIENT


@app.route("/send_heartbeat", methods=["POST"])
def send_heartbeat():
    token = request.headers["Token"]
    print("\n\nREQUEST:", request)
    print("request.data:", request.json)
    valid = auth.check_token_validity(token, server.get_client_data())
    if valid:
        print(token, " client is alive.")
        info = request.json
        # UPDATE CLIENT INFO
        server.client_info[token] = info

        response = make_response()
        response.status_code = values.OK_STATUS
    else:  # INVALID CLIENT
        response = make_response()
        response.status_code = values.INVALID_CLIENT
    return response


@app.route("/get_client_info_json", methods=["GET", "OPTIONS"])
@cross_origin(supports_credentials=True)
def send_client_info_json():
    server.set_client_heartbeat_status()
    return json.dumps(server.client_info)


@app.route("/get_server_info_json", methods=["GET"])
def send_server_info_json():
    info = server.get_server_info_as_dict()
    return json.dumps(info)


class Server:
    def __init__(self, config):
        super().__init__()
        self.config = config
        # CREATE LOCK OBJECT
        self.lock = threading.Lock()
        self.global_update = False
        self.mode = config["mode"]
        self.name = config["name"]
        self.arch = config["arch"]

        # OPEN TOKEN DATA
        if os.path.isfile(values.TOKEN_FILE):  # TOKENS FILE EXISTS
            with open(values.TOKEN_FILE, "rb") as file:
                self.tokens = pickle.load(file)
        else:
            self.tokens = {"default": 0}  # CREATE EMPTY TOKENS LIST

        # OPEN CLIENT DATA
        if os.path.isfile(values.CLIENT_DATA_FILE):  # CLIENT DATA EXISTS
            with open(values.CLIENT_DATA_FILE, "rb") as file:
                self.client_data = pickle.load(file)
        else:
            self.client_data = {}  # STORE CLIENT INFO

        self.client_data_length = {}

        self.client_info = {}

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
                except Exception as e:
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
            if self.mode == values.DETECTION_MODE:
                model = m.DetectionModel(self.config["arch"], self.config["n_classes"])
            else:
                # default is classification
                model = m.ClassificationModel(
                    self.config["arch"], self.config["n_classes"]
                )

            torch.save(model.state_dict(), self.model_path)
        else:
            self.model_path = self.config["model_path"]

        self.aggregation = self.config["aggregation"]
        # NO OF ROUNDS OF FL
        self.rounds = self.config["rounds"]
        self.total_rounds = self.config["rounds"]

        print("\nSERVER INITIALISED")

    def get_server_info_as_dict(self):
        info = {}
        info["name"] = self.name
        info["model_path"] = self.model_path
        info["client_updates_path"] = self.client_updates_path
        info["mode"] = self.mode
        info["host"] = self.HOST
        info["port"] = self.PORT
        info["architecture"] = self.arch
        info["aggregation"] = self.aggregation
        info["fl_rounds"] = self.total_rounds
        info["fl_rounds_left"] = self.rounds
        return info

    def set_client_heartbeat_status(self):
        current_time = time()
        print(self.client_info)
        for key in self.client_info.keys():
            print("CLIENT INFO TIME:", self.client_info[key])
            # CHECK TIME DIFFERENCE
            heartbeat_time = self.client_info[key]["time"]
            diff = current_time - heartbeat_time

            if (
                diff > 10 and self.client_info[key]["condition"] != "Alive"
            ):  # 10 seconds
                self.client_info[key]["condition"] = "Dead"

    def save_tokens(self):
        with open("tokens.pkl", "wb") as file:
            pickle.dump(self.tokens, file)

    def save_client_data(self):
        with open("client_data.pkl", "wb") as file:
            pickle.dump(self.client_data, file)

    def get_client_data(self):
        return self.client_data

    def get_token_list(self):
        return self.tokens

    def add_token_to_list(self, token):
        self.tokens[token] = token
        self.save_tokens()

    def server_receive_ok(self, token):
        self.lock.acquire()  # BLOCKS UNTIL LOCK IS ACQUIRED
        self.client_data[token] = LOCAL_MODEL_RECEIVED
        self.save_client_data()

        result = fl_tools.check_client_data(
            self.get_client_data(),
            self.client_data_length,
            self.aggregation,
            self.client_updates_path,
            self.model_path,
        )  # RUN WITH BLOCKING
        self.lock.release()

        if result == True:
            self.rounds -= 1
            self.global_update = True
        print("\nFL ROUNDS LEFT: ", (self.rounds))  # TEMP
        if self.rounds < 0:
            print("FL COMPLETED")
            exit()

    def client_receive_ok(self, token):
        self.lock.acquire()  # BLOCKS UNTIL LOCK IS ACQUIRED
        self.client_data[token] = GLOBAL_MODEL_SENT
        self.save_client_data()
        self.lock.release()

        if self.global_update:
            self.check_global_update()

    def set_client_data_length(self, token, length):
        self.client_data_length[token] = length

    def check_global_update(self):
        if self.global_update:
            count = 0
            for k, v in self.client_data.items():
                if v == GLOBAL_MODEL_SENT:
                    count += 1
            if count == len(self.client_data.keys()):
                self.global_update = False

    def add_client_to_list(self, token):
        self.client_data[token] = values.NEW_CLIENT
        self.save_client_data()

    def check_global_update_for_client(self, token):
        if (
            self.global_update
            and self.get_client_data()[token] == values.LOCAL_MODEL_RECEIVED
        ):
            return True
        return False


if __name__ == "__main__":
    with open("server_config.yaml") as file:
        config = yaml.safe_load(file)
    server = Server(config)
    app.run(server.HOST, port=server.PORT, threaded=True)

