"""
Starts the server. 
"""

import fl_server
import yaml
def main():
    with open('server_config.yaml') as file:
        config = yaml.safe_load(file)
    
    server = fl_server.fl_server(config)
    server.start()

if __name__ == "__main__":
    main()
