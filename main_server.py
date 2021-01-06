import socket
# import Client
import Server
import fl_server

HOST = '127.0.0.1'
PORT = 9865

# server = Server.Server()
# server.start(HOST, PORT, 5)
server = fl_server.fl_server()
server.start(HOST, PORT)

# client = Client.Client()
# client.connect(HOST, PORT)

# client.get('MODEL')