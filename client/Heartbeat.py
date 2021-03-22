import threading

class Heartbeat:
    def __init__(self, client, config) -> None:
        self.heartbeat = {}
        self.heartbeat['token'] = client.token
        self.heartbeat['total_epochs'] = config['epochs']
        self.heartbeat['mode'] = config['mode']
        self.heartbeat['send_after_epoch'] = config['send_after_epoch']
        self.heartbeat['weight_update'] = config['weight_update']
        self.heartbeat['current_epoch'] = 1
        self.heartbeat['condition'] = 'alive'
        self.heartbeat['data_length'] = 0
        self.heartbeat['status'] = 'get model'
        self.thread = None

    def set_current_epoch(self, epoch):
        self.heartbeat['current_epoch'] = epoch

    def set_status(self, status):
        self.heartbeat['status'] = status
    
    def set_condition(self, condition):
        self.heartbeat['condition'] = condition
    
    def set_data_length(self, length):
        self.heartbeat['data_length'] = length

    def scheduler(self,client):
        status = client.send_heartbeat_to_server(self.heartbeat)
        self.thread = threading.Timer(5.0, self.scheduler, (client,))
        self.thread.daemon = True
        self.thread.start()

         
