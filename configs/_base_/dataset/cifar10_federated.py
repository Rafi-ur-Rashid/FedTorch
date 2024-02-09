_base_ = "./_main_.py"
data = dict(
    dataset=dict(
        type='cifar10_federated',
        root="./data/cifar10_federated", 
        download=False, 
        num_classes=10,
        dimension=[3,32,32],
    )
)