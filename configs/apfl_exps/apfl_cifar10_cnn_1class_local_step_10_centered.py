_base_ = [
    '../_base_/model/lenet.py', # Model config
    '../_base_/dataset/cifar10.py', # Data Config
    '../_base_/optimizer/sgd.py', # Optimizer Config
    '../_base_/scheduler/convex_decay.py', # Scheduler Config
    '../_base_/checkpoint.py', # Checkpoint Config
    '../_base_/device.py', # Device Config
    '../_base_/partitioner/federated_partitioner.py', # Partitioner Config
    '../_base_/training/federated.py', # Training Config
    '../_base_/federated/apfl.py',  # Federated Learning Config
]

training=dict(
    centered=True,
    local_step=10,
)

data = dict(
    dataset=dict(
        download=True,
    )
)

federated = dict(
    sync_type='local_step',
    personal=True,
    apfl_alpha=0.25,
    apfl_adaptive_alpha=False
)

device=dict(
    type='cpu'
)
