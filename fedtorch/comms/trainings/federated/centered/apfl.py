# -*- coding: utf-8 -*-
import time
from copy import deepcopy
import numpy as np

import torch

from fedtorch.components.scheduler_builder import adjust_learning_rate
from fedtorch.components.dataset_builder import load_data_batch
from fedtorch.comms.utils.flow_utils import (get_current_epoch, 
                                             get_current_local_step, 
                                             is_sync_fed,
                                             alpha_update)
from fedtorch.comms.utils.eval import inference, inference_personal
from fedtorch.comms.utils.eval_centered import do_validate_centered, log_validation_centered
from fedtorch.comms.algorithms.federated import fedavg_aggregation_centered, set_online_clients_centered
from fedtorch.logs.logging import (log, 
                                   logging_sync_time,
                                   logging_load_time,
                                   logging_globally)
from fedtorch.logs.meter import define_local_training_tracker

def train_and_validate_apfl_centered(Clients, Server):
    log('start training and validation with Federated setting in a centered way.')

    # For Sparsified SGD
    tracker = define_local_training_tracker()
    start_global_time = time.time()
    tracker['start_load_time'] = time.time()
    log('enter the training.')

    # Number of communication rounds in federated setting should be defined
    for n_c in range(Server.cfg.federated.num_comms):
        Server.cfg.rounds_comm += 1
        Server.cfg.local_index += 1
        Server.cfg.quant_error = 0.0
        
        # Preset variables for this round of communication
        Server.zero_grad()
        Server.reset_tracker(Server.local_val_tracker)
        Server.reset_tracker(Server.global_val_tracker)
        if Server.cfg.federated.personal:
            Server.reset_tracker(Server.local_personal_val_tracker)
            Server.reset_tracker(Server.global_personal_val_tracker) 

        # Configuring the devices for this round of communication
        log("Starting round {} of training".format(n_c+1))
        online_clients = set_online_clients_centered(Server.cfg)
        
        for oc in online_clients:
            Clients[oc].model.load_state_dict(Server.model.state_dict())
            Clients[oc].cfg.rounds_comm = Server.cfg.rounds_comm
            local_steps = 0
            is_sync = False

            do_validate_centered(Clients[oc].cfg, Server.model, Server.criterion, Server.metrics, Server.optimizer,
                 Clients[oc].train_loader, Server.global_val_tracker, val=False)
            if Server.cfg.federated.personal:
                do_validate_centered(Clients[oc].cfg, Server.model, Server.criterion, Server.metrics, Server.optimizer,
                    Clients[oc].val_loader, Server.global_personal_val_tracker, val=True)

            while not is_sync:
                for _input, _target in Clients[oc].train_loader:
                    local_steps += 1
                    Clients[oc].model.train()

                    # update local step.
                    logging_load_time(tracker)
                    
                    # update local index and get local step
                    Clients[oc].cfg.local_index += 1
                    Clients[oc].cfg.local_data_seen += len(_target)
                    get_current_epoch(Clients[oc].cfg)
                    local_step = get_current_local_step(Clients[oc].cfg)

                    # adjust learning rate (based on the # of accessed samples)
                    lr = adjust_learning_rate(Clients[oc].cfg, Clients[oc].optimizer, Clients[oc].scheduler)

                    # load data
                    _input, _target = load_data_batch(Clients[oc].cfg, _input, _target, tracker)
        
                    # Skip batches with one sample because of BatchNorm issue in some models!
                    if _input.size(0)==1:
                        is_sync = is_sync_fed(Clients[oc].cfg)
                        break

                    # inference and get current performance.
                    Clients[oc].optimizer.zero_grad()
                    
                    loss, _ = inference(Clients[oc].model, Clients[oc].criterion, Clients[oc].metrics, _input, _target)

                    # compute gradient and do local SGD step.
                    loss.backward()

                    Clients[oc].optimizer.step(
                        apply_lr=True,
                        apply_in_momentum=Clients[oc].cfg.training.in_momentum, apply_out_momentum=False
                    )

                    Clients[oc].optimizer.zero_grad()
                    Clients[oc].optimizer_personal.zero_grad()
                    loss_personal, _ = inference_personal(Clients[oc].model_personal, Clients[oc].model,
                                                                    Clients[oc].cfg.federated.apfl_alpha, Clients[oc].criterion, 
                                                                    Clients[oc].metrics, _input, _target)
                    # compute gradient and do local SGD step.
                    loss_personal.backward()
                    Clients[oc].optimizer_personal.step(
                        apply_lr=True,
                        apply_in_momentum=Clients[oc].cfg.training.in_momentum, 
                        apply_out_momentum=False
                    )

                    # update alpha
                    if Clients[oc].cfg.federated.apfl_adaptive_alpha and local_steps == 1:
                        Clients[oc].cfg.federated.apfl_alpha = alpha_update(Clients[oc].model, Clients[oc].model_personal,
                                                                                Clients[oc].cfg.federated.apfl_alpha, lr)          
                        log("New alpha is:{}".format(Clients[oc].cfg.federated.apfl_alpha), Clients[oc].cfg.graph.debug)

                    # reset load time for the tracker.
                    tracker['start_load_time'] = time.time()
                    # model_local = deepcopy(model_client)
                    is_sync = is_sync_fed(Clients[oc].cfg)
                    if is_sync:
                        break



            do_validate_centered(Clients[oc].cfg, Clients[oc].model, Clients[oc].criterion, Clients[oc].metrics, Clients[oc].optimizer,
                 Clients[oc].train_loader, Server.local_val_tracker, val=False, personal=True,
                  model_personal=Clients[oc].model_personal,alpha=Clients[oc].cfg.federated.apfl_alpha)
            if Server.cfg.federated.personal:
                do_validate_centered(Clients[oc].cfg, Clients[oc].model, Clients[oc].criterion, Clients[oc].metrics, Clients[oc].optimizer,
                 Clients[oc].val_loader, Server.local_personal_val_tracker, val=True, personal=True,
                  model_personal=Clients[oc].model_personal, alpha=Clients[oc].cfg.federated.apfl_alpha)
            # Sync the model server based on model_clients
            tracker['start_sync_time'] = time.time()
            Server.cfg.global_index += 1

            logging_sync_time(tracker)
        
        fedavg_aggregation_centered(Clients, Server, online_clients)
        # Log local performance
        log_validation_centered(Server.cfg, Server.local_val_tracker, val=False, personal=True)
        if Server.cfg.federated.personal:
            log_validation_centered(Server.cfg, Server.local_personal_val_tracker, val=True, personal=True,  model_name="Personal")
            

        # Log server performance
        log_validation_centered(Server.cfg, Server.global_val_tracker, val=False, local=False)
        if Server.cfg.federated.personal:
            log_validation_centered(Server.cfg, Server.global_personal_val_tracker, val=True, local=False)


        # logging.
        logging_globally(tracker, start_global_time)
        
        # reset start round time.
        start_global_time = time.time()

        # validate the model at the server
        # if cfg.graph.rank == 0:
        #     do_test(cfg, model_server, optimizer, criterion, metrics, test_loader)
        # do_validate_test(cfg, model_server, optimizer, criterion, metrics, test_loader)
    return