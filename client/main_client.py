import socket
import torch
import torchvision
import yaml

import train
import Client

import numpy as np

import model as m
import data_loader as dl
import transforms as t
import traceback

from time import time, sleep
from torchvision import datasets, transforms
from torch import nn, optim
from copy import deepcopy
from torch.utils.data import Subset

import threading
from Heartbeat import Heartbeat

import sys
sys.path.append("../")
import values

def check_model_similarity(model1, model2):
    """
    Check if two models are the same.
    """
    flag = 0
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            continue
        else:
            flag += 1
            break
    if flag > 0:
        print("\nMODELS SAME", flag)
    else:
        print("\nMODELS NOT SAME")


def main():
    with open("client_config.yaml") as file:
        config = yaml.safe_load(file)
    epochs = config["epochs"]


    lr = config["lr"]
    mode = config["mode"]
    send_after_epoch = config["send_after_epoch"]
    weight_update = config["weight_update"]

    if config["device"]:
        device = torch.device(config["device"])
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))
        print("Memory Usage:")
        print("Allocated:", round(torch.cuda.memory_allocated(0) / 10243, 1), "GB")
        print("Cached:   ", round(torch.cuda.memory_reserved(0) / 10243, 1), "GB")

    labels = config["labels"]
    n_classes = config["n_classes"]
    assert len(labels) == n_classes, "ERR: Number of classes should match labels."

    # GET MODEL
    if mode == "detection":
        train_loader, test_loader = dl.get_detection_dataset(
            train_csv_file=config["train_csv_file"],
            train_path=config["train_image_dir"],
            train_transform=t.transform_train_detection,
            train_bs=config["train_batch_size"],
            test_csv_file=config["test_csv_file"],
            test_path=config["test_image_dir"],
            test_transform=t.transform_test_detection,
            test_bs=config["test_batch_size"],
            debug=False,
        )
        model = m.DetectionModel(config["arch"], n_classes)
    else:
        # default is classification
        train_loader, test_loader = dl.get_classification_dataset(
            train_csv_file=config["train_csv_file"],
            train_path=config["train_image_dir"],
            train_transform=t.transform_train,
            train_labels=labels,
            train_bs=config["train_batch_size"],
            test_csv_file=config["test_csv_file"],
            test_path=config["test_image_dir"],
            test_transform=t.transform_test,
            test_labels=[],
            test_bs=config["test_batch_size"],
            debug=True,
        )
        model = m.ClassificationModel(config["arch"], n_classes)

    global client
    client = Client.Client(config)
    print("\nCONNECTED TO SERVER")
    client.data_length = len(train_loader.dataset)

    global heartbeat 
    heartbeat = Heartbeat(client, config)
    heartbeat.set_data_length(client.data_length)
    heartbeat.set_status(values.client_receiving_model)
    

    result = client.get()

    current_epoch_for_client = client.get_current_epoch_for_client()
    #logic to resume if send after epoch is not 
    current_epoch_for_client = current_epoch_for_client - (current_epoch_for_client % send_after_epoch)

    heartbeat.set_current_epoch(current_epoch_for_client)

    heartbeat.scheduler(client)

    if result == False: # COULD NOT GET MODEL
        print('\nCOULD NOT GET MODEL FROM SERVER')
        exit()
    print("\nSAVED GLOBAL MODEL")


    model.load_state_dict(torch.load(client.model_folder + "/" + client.model_name))

    # MAKE A COPY OF INITIAL MODEL WEIGHTS IF WEIGHT UPDATE IS UGA
    if weight_update == "uga": 
        initial_model_weights = deepcopy(model.state_dict())

    criterion = torch.nn.BCEWithLogitsLoss()
    optim_params = [{"params": model.parameters(), "lr": lr}]
    if config["optimizer"] == "SGD":
        optimizer = optim.SGD(optim_params)
    elif config["optimizer"] == "Adam":
        optimizer = optim.Adam(optim_params)
    else:
        # default
        optimizer = optim.Adam(optim_params)

    time0 = time()

    
    for epoch in range(current_epoch_for_client, epochs):
        heartbeat.set_current_epoch(epoch)
        heartbeat.set_status(values.client_receive_model_success)
        print("Epoch {}/{}".format(epoch, epochs - 1))
        print("-" * 10)
        running_loss = 0
        # TRAIN
        if mode == "detection":
            running_loss += train.detection_train(
                model, train_loader, optimizer, device
            )
        else:
            # classification
            if weight_update == "normal" or (epoch+1) % send_after_epoch != 0:
                # TRAIN
                running_loss += train.classification_train(
                    model, train_loader, optimizer, criterion, device
                )

            elif weight_update == "uga" and (epoch+1) % send_after_epoch == 0:
                running_loss += train.train_last_uga(
                    initial_model_weights, model, train_loader, optimizer, criterion, device
                )

        print(
            "Epoch {} - Training loss: {}".format(
                epoch, running_loss / len(train_loader)
            )
        )
        temp1 = deepcopy(model)

        # SAVE MODEL AFTER EVERY EPOCH
        torch.save(model.state_dict(), client.model_folder + "/" + client.model_name)

        if (epoch+1) % send_after_epoch == 0: # SEND TO SERVER
            heartbeat.set_status(values.client_sending_model)
            result = client.send()
            if result == False:
                print('\nCOULD NOT SEND TO SERVER')
                exit()

            # CHECK FOR UPDATES
            heartbeat.set_status(values.client_waiting)
            result = client.check_update()
            while result == False:
                print("\nSERVER HAS NOT UPDATED GLOBAL")
                sleep(1)
                result = client.check_update()

            heartbeat.set_status(values.client_receiving_model)
            result = client.get() # GET UPDATED MODEL
            if result == False:
                print('\nCOULD NOT GET MODEL FROM SERVER')
                exit()

            model.load_state_dict(torch.load(client.model_folder + "/" + client.model_name))
            check_model_similarity(temp1, model)

            # RESET INTIAL WEIGHTS FOR UGA TO NEW MODEL
            if weight_update == "uga": 
                initial_model_weights = deepcopy(model.state_dict()) 

    print("\nTraining Time (in minutes) =", (time() - time0) / 60)


if __name__ == "__main__":
    try:
        main()
        heartbeat.set_condition("Completed")
        heartbeat.set_status("FL Completed")
        client.send_heartbeat_to_server(heartbeat.heartbeat)
        exit(0)
    except KeyboardInterrupt as e:
        traceback.print_exc()
        heartbeat.set_exception("KeyboardInterrupt")
        heartbeat.set_condition("Dead")
        client.send_heartbeat_to_server(heartbeat.heartbeat)
    except Exception as e:
        traceback.print_exc()
        heartbeat.set_exception(str(e))
        heartbeat.set_condition("Dead")
        client.send_heartbeat_to_server(heartbeat.heartbeat)
        exit(1)
        

    

