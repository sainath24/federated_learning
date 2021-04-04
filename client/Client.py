import socket
from time import sleep
from torch.cuda.memory import empty_cache
import tqdm
from werkzeug.datastructures import Headers
import yaml
import os
import pickle
import json

import re
import traceback

import select
import requests

import sys

sys.path.append("../")
import values
from values import TOKEN_BUFFER_SIZE, BUFFER_SIZE, SEPARATOR

# QUERIES
# MODEL - get latest model
# CONFIG - get latest config
# UPDATE_MODEL - send updated model

# TOKEN_BUFFER_SIZE = 16
# BUFFER_SIZE = 4096
# SEPARATOR = '&'

# MODEL_NAME = 'model.pth'
# CONFIG_NAME = 'config.yaml'
MAX_RETRY = 5


class Client:
    def __init__(self, config):
        super().__init__()
        print("\nINITIALISING CLIENT...")
        print("CONFIG:")
        print(config)
        self.config = config
        self.HOST = self.config["HOST"]
        self.PORT = self.config["PORT"]
        self.token = ""

        # MODEL FOLDER
        try:
            self.model_folder = self.config["model_folder"]
            if not os.path.isdir(self.model_folder):
                print("\nFOLDER UNAVAILABLE, CREATING FOLDER...")
                try:
                    os.makedirs(self.model_folder)
                except:
                    print("\nUNABLE TO CREATE FOLDER")
                    if not os.path.isdir("MODEL"):
                        print("\nCREATING DEFAULT PATH /MODEL...")
                        os.mkdir("MODEL")
                    else:
                        print("\nUSING DEFAULT PATH: /MODEL")
                    self.model_folder = "MODEL"

        except Exception as e:
            print(e)
            # print('\nPATH TO MODEL NOT FOUND...')
            if not os.path.isdir("MODEL"):
                print("\nCREATING DEFAULT PATH /MODEL...")
                os.mkdir("MODEL")
            else:
                print("\nUSING DEFAULT PATH: /MODEL")
            self.model_folder = "MODEL"

        # MODEL NAME
        try:
            self.model_name = self.config["model_name"]
            print("MODEL NAME: ", self.model_name)
        except Exception as e:
            print("\nMODEL NAME NOT FOUND, USING DEFAULT NAME model.pth")
            self.model_name = "model.pth"

        self.url = self.HOST + ":" + self.PORT

        self.data_length = -1

        self.handle_token()
        print("\nCLIENT INITIALISED")

    def is_token(self, token):
        check = re.fullmatch(values.TOKEN_REGEX, token)
        if check:
            return True
        return False

    def received_token(self):
        url = self.url + "/received_token"
        header = {"Token": self.token, "Datalength": str(self.data_length)}
        response = requests.get(url, headers=header)
        if response.headers["Token"] == values.receive_token_valid:
            return True
        return False

    def request_token(self):
        url = self.url + "/register_client"
        header = {"Token": values.REQUIRES_TOKEN}
        response = requests.get(url, headers=header)
        token = response.text
        if self.is_token(token):  # RECEIVED TOKEN
            self.token = token
            self.save_token(token)
            result = self.received_token()
            return result
        return False

    def save_token(self, token):
        with open("token.pkl", "wb") as file:
            pickle.dump(token, file)  # SAVE TOKEN
        file.close()

    def handle_token(self):
        result = False
        try:
            if os.path.isfile("token.pkl"):  # TOKEN IS PRESENT
                with open("token.pkl", "rb") as file:
                    self.token = pickle.load(file)
                    result = True
            else:  # NEW CLIENT, REQUIRES TOKEN
                result = self.request_token()
            return result
        except:
            traceback.print_exc()
        return False

    def received_model(self):
        url = self.url + "/model_received"
        header = {"Token": self.token}
        response = requests.get(url, headers=header)
        if response.status_code == values.OK_STATUS:
            return True
        return False

    def get(self):
        result = False
        for i in range(MAX_RETRY):
            try:
                url = self.url + "/get_model"
                header = {"Token": self.token, "Datalength": str(self.data_length)}

                response = requests.get(url, headers=header)
                if (
                    response.status_code == values.OK_STATUS
                ):  # CLIENT IS AUTHENTICATED, GET THE MODEL FILE
                    file_size = int(response.headers["Filesize"])
                    model_file = response.content
                    if len(model_file) == file_size:  # MODEL IS OK
                        path = os.path.join(self.model_folder, self.model_name)
                        open(path, "wb").write(model_file)
                        self.received_model()
                        result = True

            except Exception as e:
                print("\nEXCEPTION IN get: ", e)
                traceback.print_exc()
                result = False
            if result == True:
                break
            else:
                print(values.get_failed_retry)
                if i + 1 != MAX_RETRY:
                    sleep(10)

        if result == False:
            print(values.get_failed)
        return result

    def send(self):  # SEND UPDATED MODEL
        result = False
        for i in range(MAX_RETRY):
            try:
                url = self.url + "/send_model"
                header = {"Token": self.token, "Datalength": str(self.data_length)}

                filename = self.model_name
                path = os.path.join(self.model_folder, self.model_name)
                filesize = os.path.getsize(path)
                header["Filesize"] = str(filesize)

                model_file = {"model": open(path, "rb")}

                response = requests.post(url, headers=header, files=model_file)
                if response.status_code == values.OK_STATUS:  # SEREVR RECEIVED MODEL
                    result = True

            except Exception as e:
                print("\nEXCEPTION in send: ", e)
                traceback.print_exc()
                result = False

            if result == True:
                break
            else:
                print(values.send_failed_retry)
                if i + 1 != MAX_RETRY:
                    sleep(10)

        if result == False:
            print(values.send_failed)
        return result

    def check_update(self):
        url = self.url + "/check_update"
        header = {"Token": self.token}
        response = requests.get(url, headers=header)
        if response.text == values.MODEL_UPDATE_AVAILABLE:
            return True
        else:
            return False

    def send_heartbeat_to_server(self, msg):
        try:
            url = self.url + "/send_heartbeat"
            header = {"Token": self.token}
            print("msg:", msg)
            response = requests.post(url, headers=header, json=msg)
            if response.text == values.OK_STATUS:
                return True
            else:
                return False
        except Exception as e:
            print("\nEXCEPTION in send: ", e)
            traceback.print_exc()

    def get_current_epoch_for_client(self):
        url = self.url + "/get_current_epoch_for_client"
        header = {"Token": self.token}
        response = requests.get(url, headers=header)
        if response.status_code == values.OK_STATUS:
            return int(response.headers["current_epoch"])
        else:
            return 1

