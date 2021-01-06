import Server
import threading

class fl_server:
    def __init__(self, rounds = 5, max_client = 5) -> None:
        super().__init__()
        self.rounds = rounds
        self.max_clients = max_client
        self.current_client = 0
        self.server = Server.Server()
        # ACCESS CLIENT DATA WITH self.server.client_data

    def update_central(self):
        client_data = self.server.get_client_data()
        # MAKE A LIST OF MODELS USED FOR CENTRAL UPDATE
        models = []
        for k,v in client_data.items():
            if v==2:
                models.append(str(k) + '.pth')
                
        # TODO: UPDATE CENTRAL MODEL using models list
        # UPDATE MODELS
        self.rounds -=1
        if self.rounds<=0: # FL IS OVER
            print('\nFEDERATED LEARNING OVER')
            exit()

    def check_client_data(self):
        waiting_clients = 0
        try:
            data = self.server.get_client_data()
            total_clients = len(data.keys())
            for k,v in data.items():
                if v == 2:
                    waiting_clients+=1
            # TODO: CHECK CLIENTS
            # CHECK IF YOU NEED TO UPDATE CENTRAL MODEL

        except Exception as e:
            print('check_client_data: ', e)

    def start_server(self, host, port):
        try:
            self.server.start(host, port, self.max_clients, self.check_client_data)
        except Exception as e:
            print('\nEXCEPTING in start_server: ' ,e)

    def start(self,host, port):
        # START SERVER ON SEPARATE THREAD
        server_thread = threading.Thread(target=self.start_server, args=[host, port])
        server_thread.start()







        


        


        