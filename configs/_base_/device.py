device = dict(
    is_distributed=True,
    hostfile='hostfile',
    dist_backend='mpi',
    type='cuda', # or 'cpu', or 'mps
    num_nodes=1,
    num_clients=10,
)