_base_ = "./_main_.py"
federated = dict(
    type='perfedavg',
    personal=True,
    perfedavg_beta=0.001, # The beta parameter in PerFedAvg algorithm. See paper: https://arxiv.org/pdf/2002.07948.pdf
)