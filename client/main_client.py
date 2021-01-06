import socket
import Client
# import Server

HOST = '127.0.0.1'
PORT = 9865

# server = Server.Server()
# server.start(HOST, PORT, 5)

client = Client.Client()
client.connect(HOST, PORT)

client.get('MODEL')

client.connect(HOST, PORT)
client.send('UPDATE_MODEL')