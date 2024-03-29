# -*- coding: utf-8 -*-
import numpy as np
import torch

from fedtorch.comms.utils.flow_utils import quantize_tensor, dequantize_tensor

def fedavg_aggregation_centered(OnlineClients, Server, online_clients, lambda_weight=None):
    """Aggregate gradients for federated learning using FedAvg algorithm.

    Each local model first gets the difference between current model and
    previous synchronized model, and then all-reduce these difference by SUM.

    """
    Server.optimizer.zero_grad()
    num_online_clients = len(online_clients)
    if lambda_weight is None:
        # rank_weight =  OnlineClient.cfg.data.num_samples_per_epoch / OnlineClient.cfg.data.train_dataset_size
        rank_weight =  [1.0 / num_online_clients] * Server.cfg.graph.n_nodes
    else:
        #TODO: This is experimental. Test it.
        rank_weight = lambda_weight * Server.cfg.graph.n_nodes / num_online_clients

    for o in online_clients:
        for i, (server_param, client_param) in enumerate(zip(Server.model.parameters(), OnlineClients[o].model.parameters())):
            # get model difference.
            param_diff = (server_param.data - client_param.data) * rank_weight[o]
            
            if Server.cfg.federated.type == 'fedadam':
                # Based on https://arxiv.org/abs/2003.00295
                OnlineClients[o].cfg.federated.fedadam_v[i] = OnlineClients[o].cfg.federated.fedadam_beta * OnlineClients[o].cfg.federated.fedadam_v[i] + (1-OnlineClients[o].cfg.federated.fedadam_beta) * torch.norm(param_diff)
                param_diff /= (np.sqrt(OnlineClients[o].cfg.federated.fedadam_v[i]) + OnlineClients[o].cfg.federated.fedadam_tau)
            
            if Server.cfg.federated.quantized:
                grad_q, q_info = quantize_tensor(param_diff, num_bits= Server.cfg.federated.quantized_bits, adaptive=True)
                grad_deq = dequantize_tensor(grad_q, q_info)
                
                # with torch.no_grad():
                server_param.grad.data.add_(grad_deq)
            else:
                server_param.grad.data.add_(param_diff)

    Server.optimizer.step(
        apply_lr=False,
        scale=Server.cfg.lr.lr_scale_at_sync,
        apply_in_momentum=False,
        apply_out_momentum=Server.cfg.training.out_momentum,
    )
    return 