import os
try:
    os.remove("client_data.pkl")
except:
    pass
try:
    os.remove("tokens.pkl")
except:
    pass
try:
    os.removedirs("CLIENT_UPDATES")
except:
    pass
os.system("python main_server.py")