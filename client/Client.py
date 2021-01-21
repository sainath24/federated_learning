import socket
import tqdm
import yaml
import os
import pickle

import re
import traceback

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

        # CREATE SOCKET
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("\nCLIENT INITIALISED")

    def is_token(self, token):
        check = re.fullmatch(values.TOKEN_REGEX, token)
        if check:
            return True
        return False

    def request_token(self):
        self.s.send(values.REQUIRES_TOKEN.encode())
        response = self.s.recv(
            TOKEN_BUFFER_SIZE
        ).decode()  # receive response from server
        if self.is_token(response):  # RECEIVED TOKEN
            self.save_token(response)
            return True
        return False

    def send_token(self, token):
        self.s.send(token.encode())
        response = self.s.recv(TOKEN_BUFFER_SIZE)  # receive response from server
        if response == values.receive_token_invalid:
            try:
                os.remove("token.pkl")
                print(values.client_invalid_token)
            except:
                traceback.print_exc()
            return False
        elif response == values.receive_token_valid:  # TOKEN AUTHENTICATED
            return True

    def save_token(self, token):
        with open("token.pkl", "wb") as file:
            pickle.dump(token, file)  # SAVE TOKEN
        file.close()

    def handle_token(self):
        try:
            if os.path.isfile("token.pkl"):  # TOKEN IS PRESENT
                with open("token.pkl", "rb") as file:
                    token = pickle.load(file)
                result = self.send_token(token)
            else:  # NEW CLIENT, REQUIRES TOKEN
                result = self.request_token()
            return result
        except:
            traceback.print_exc()
        return False

    def connect(self):
        result = True
        for i in range(MAX_RETRY):
            try:
                self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.s.connect((self.HOST, self.PORT))
                result = self.handle_token()
                if result:
                    print(values.client_connection_success)
                    break
                else:
                    print(values.client_connection_fail_retry)
            except Exception as e:
                print("\nEXCEPTION IN CONNECT: ", e)
                traceback.print_exc()
                result = False
        if result == False:
            print(values.client_connection_fail)

        return result

    def get(self):
        result = False
        for i in range(MAX_RETRY):
            try:
                # OK RESPONSE
                data = self.s.recv(BUFFER_SIZE).decode()
                print("\nRESPONSE FROM SERVER: ", data)
                if data == values.no_update_available:
                    print("\nGOT NOT UPDATED RESPONSE\n")
                    result = False

                else:
                    # UPDATE AVAILABLE
                    try:
                        filename, filesize = data.split(SEPARATOR)
                        filesize = int(filesize)
                        path = os.path.join(self.model_folder, self.model_name)
                        self.s.sendall(values.metadata_valid.encode())
                        result = True
                    except:  # INVALID METADATA
                        self.s.sendall(values.metadata_invalid.encode())
                        self.s.close()
                        result = False

                    if result == True:
                        print("\nRECEIVED INFO: ", filename, " SIZE: ", filesize)
                        # RECEIVE FILE
                        progress = tqdm.tqdm(range(filesize), "RECEIVING " + filename)
                        p = 0
                        with open(path, "wb") as file:
                            while p != filesize:
                                data = self.s.recv(BUFFER_SIZE)
                                if not data:
                                    break
                                file.write(data)
                                p += len(data)
                                progress.update(len(data))

                        if p == filesize:  # SEND OK
                            self.s.sendall(values.client_receive_model_success.encode())
                            print(values.client_receive_model_success)
                        else:
                            result = False

            except Exception as e:
                print("\nEXCEPTION IN get: ", e)
                traceback.print_exc()
                result = False
            self.s.close()
            if result == True:
                break
            else:
                print(values.get_failed_retry)
                if i + 1 != MAX_RETRY:
                    self.connect()

        if result == False:
            print(values.get_failed)
        return result

    def send(self):  # SEND UPDATED MODEL
        result = False
        for i in range(MAX_RETRY):
            try:
                filename = self.model_name
                path = os.path.join(self.model_folder, self.model_name)
                filesize = os.path.getsize(path)
                print("CLIENT MODEL PATH: ", path)
                self.s.send(f"{filename}{SEPARATOR}{filesize}".encode())

                response = self.s.recv(
                    TOKEN_BUFFER_SIZE
                ).decode()  # GET METADATA RESPONSE FROM SERVER
                if response == values.metadata_valid:  # VALID METADATA
                    progress = tqdm.tqdm(
                        range(filesize), "SENDING UPDATED MODEL TO SERVER"
                    )
                    p = 0
                    with open(path, "rb") as file:
                        while True:
                            read = file.read(BUFFER_SIZE)
                            if not read:
                                break
                            self.s.sendall(read)
                            p += len(read)
                            progress.update(len(read))

                    if (
                        p != filesize
                    ):  # ERROR IN SEND, FULL FILE HAS NOT BEEN SENT, RETRY
                        result = False
                        print(values.send_model_fail)

                elif response == values.metadata_invalid:
                    print(values.metadata_invalid)
                    result = False
                else:  # INVALID RESPONSE FROM CLIENT
                    print(values.client_invalid_response)
                    result = False

                if result == True:  # RECEIVE OK
                    response = self.s.recv(TOKEN_BUFFER_SIZE).decode()
                    print("\nRESPONSE: ", response)

            except Exception as e:
                print("\nEXCEPTION in send: ", e)
                traceback.print_exc()
                # self.s.close()
                result = False

            self.s.close()
            if result == True:
                break
            else:
                print(values.send_failed_retry)
                if i + 1 != MAX_RETRY:
                    self.connect()

        if result == False:
            print(values.send_failed)
        return result
