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
    for x in os.listdir("CLIENT_UPDATES"):
        os.remove("CLIENT_UPDATES/"+x)
    os.removedirs("CLIENT_UPDATES")
except:
    pass
if not os.path.exists("GLOBAL"):
    os.makedirs("GLOBAL/")
os.system("python Server.py")