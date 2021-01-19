import socket
try:
    x = socket.getaddrinfo("4.tcp.ngrok.io", 11511)
    print(x)
except socket.gaierror as e:
    print(e)
