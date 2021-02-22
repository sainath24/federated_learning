import os
try:
    os.remove("token.pkl")
except:
    pass
try:
    os.removedirs("CLIENT_MODEL")
except:
    pass
os.system("python main_client.py")