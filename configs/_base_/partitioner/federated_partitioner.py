partitioner=dict(
    type='FederatedPartitioner',
    dataset_name=None,
    data=None,
    shuffle=False,
    sizes=None,
    distributed=True,
    sensitive_feature=None, 
    num_classes=None, 
    unbalanced=False,
    num_class_per_client=1, 
    dirichlet=False
)