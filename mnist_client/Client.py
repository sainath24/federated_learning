import socket
import tqdm
import yaml
import os
import pickle

# QUERIES
# MODEL - get latest model
# CONFIG - get latest config
# UPDATE_MODEL - send updated model

BUFFER_SIZE = 4096
SEPARATOR = '&'

# MODEL_NAME = 'model.pth'
# CONFIG_NAME = 'config.yaml'

class Client:
    def __init__(self):
        super().__init__()
        print('\nINITIALISING CLIENT...')
        with open('client_config.yaml') as file:
            yaml_data = yaml.safe_load(file)
        
        # MODEL FOLDER
        try:
            self.model_folder = yaml_data['model_folder']
            if not os.path.isdir(self.model_folder):
                print('\nFOLDER UNAVAILABLE, CREATING FOLDER...')
                try:
                    os.makedirs(self.model_folder)
                except:
                    print('\nUNABLE TO CREATE FOLDER')
                    if not os.path.isdir('MODEL'):
                        print('\nCREATING DEFAULT PATH /MODEL...')
                        os.mkdir('MODEL')
                    else:
                        print('\nUSING DEFAULT PATH: /MODEL')
                    self.model_folder = 'MODEL'
    
        except Exception as e:
            print(e)
            # print('\nPATH TO MODEL NOT FOUND...')
            if not os.path.isdir('MODEL'):
                print('\nCREATING DEFAULT PATH /MODEL...')
                os.mkdir('MODEL')
            else:
                print('\nUSING DEFAULT PATH: /MODEL')
            self.model_folder = 'MODEL'

        # MODEL NAME
        try:
            self.model_name = yaml_data['model_name']
            print('MODEL NAME: ', self.model_name)
        except Exception as e:
            print('\nMODEL NAME NOT FOUND, USING DEFAULT NAME model.pth')
            self.model_name = 'model.pth'

        # CREATE SOCKET
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print('\nCLIENT INITIALISED')
    

    def connect(self, host, port):
        try:
            self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.s.connect((host, port))
            if os.path.isfile('token.pkl'): # TOKEN IS PRESENT
                with open('token.pkl', 'rb') as file:
                    data = pickle.load(file)
                    self.s.send(data.encode())
                    status = self.s.recv(BUFFER_SIZE) # RECEIVE TOKEN_OK from server
            else: # NEW CLIENT, REQUEST TOKEN
                query = 'TOKEN'
                self.s.send(query.encode())
                token = self.s.recv(BUFFER_SIZE).decode()
                with open('token.pkl', 'wb') as file:
                    pickle.dump(token, file) # SAVE TOKEN FOR FUTURE USE

            print('\nSUCCESSFULLY CONNECTED')
        except Exception as e:
            print('\nEXCEPTION IN CONNECT: ', e)
            return False
        return True

    def get(self, query):
        # SEND QUERY
        print('\nQUERY: ' + query)
        try:
            self.s.send(query.encode())

            print('SENT QUERY: ', query)

            # OK RESPONSE
            data = self.s.recv(BUFFER_SIZE).decode()
            print('\nRESPONSE FROM SERVER: ', data)
            if data == 'NOT_UPDATED':
                print('\nGOT NOT UPDATED RESPONSE\n')
                return False
            
            #RECEIVE DATA ON FILE
            # data = self.s.recv(BUFFER_SIZE).decode()

            filename, filesize = data.split(SEPARATOR)
            filesize = int(filesize)
            path = os.path.join(self.model_folder, self.model_name)

            print('\nRECEIVED INFO: ', filename, ' SIZE: ', filesize)

            #RECEIVE FILE
            progress = tqdm.tqdm(range(filesize), "RECEIVING " + query)
            with open(path, 'wb') as file:
                while True:
                    data = self.s.recv(BUFFER_SIZE)
                    if not data:
                        break
                    file.write(data)
                    progress.update(len(data))
        except Exception as e:
            print('\nEXCEPTION IN get: ', e)
            # self.s.close()
            return False
        # self.s.close()
        return True

    def send(self, query): #SEND UPDATED MODEL
        try:
            self.s.send(query.encode())
            print('SENT QUERY: ', query)

            filename = self.model_name
            path = os.path.join(self.model_folder, self.model_name)
            filesize = os.path.getsize(path)
            print('CLIENT MODEL PATH: ', path)
            self.s.send(f"{filename}{SEPARATOR}{filesize}".encode())
            progress = tqdm.tqdm(range(filesize), "SENDING UPDATED MODEL TO SERVER")
            with open(path, 'rb') as file:
                while True:
                    read = file.read(BUFFER_SIZE)
                    if not read:
                        break
                    self.s.sendall(read)
                    progress.update(len(read))
        except Exception as e:
            print('\nEXCEPTION in send: ', e)
            # self.s.close()
            return False
        # self.s.close()
        return True
