![FedTorch Logo](./logo.png)
--------------------------------------------------------------------------------

FedTorch is an open-source Python package for distributed and federated training of machine learning models using [PyTorch distributed API](https://pytorch.org/docs/stable/distributed.html). Various algorithms for federated learning and local SGD are implemented for benchmarking and research, including our own proposed methods:
* [Redundancy Infused SGD (RI-SGD)](http://proceedings.mlr.press/v97/haddadpour19a.html) ![official](https://img.shields.io/badge/code-Official-green)
* [Local SGD with Adaptive Synchoronization (LUPA-SGD)](https://papers.nips.cc/paper/2019/hash/c17028c9b6e0c5deaad29665d582284a-Abstract.html)  ![official](https://img.shields.io/badge/code-Official-green)
* [Adaptive Personalized Federated Learning (APFL)](https://arxiv.org/abs/2003.13461) ![official](https://img.shields.io/badge/code-Official-green)
* [Distributionally Robust Federated Learning (DRFA)](https://papers.nips.cc/paper/2020/file/ac450d10e166657ec8f93a1b65ca1b14-Paper.pdf) ![official](https://img.shields.io/badge/code-Official-green)
* [Federated Learning with Gradient Tracking and Compression (FedGATE and FedCOMGATE)](https://arxiv.org/abs/2007.01154) ![official](https://img.shields.io/badge/code-Official-green)

And other common algorithms such as:
* [FedAvg](http://proceedings.mlr.press/v54/mcmahan17a.html)
* [SCAFFOLD](http://proceedings.mlr.press/v119/karimireddy20a.html)
* [Qsparse Local SGD](https://ieeexplore.ieee.org/abstract/document/9057579)
* [AFL](https://arxiv.org/abs/1902.00146)
* [FedProx](https://arxiv.org/abs/1812.06127)
* and more ...

We are actively trying to expand the library to include more training algorithms as well.

## NEWS
Recent updates to the package:
* **Paper Accepted** (01/22/2021): Our paper titled [`Federated Learning with Compression: Unified Analysis and Sharp Guarantees`](https://arxiv.org/abs/2007.01154) accepted to [AISTAT 2021](https://aistats.org/aistats2021/)🎉
* **Public Release** (01/17/2021): We are releasing the package to the public with a docker image

## Installation
First you need to clone the repo into your computer:
```cli
git clone https://github.com/MLOPTPSU/FedTorch.git
```

This package is built based on PyTorch Distributed API. Hence, it could be run with any supported distributed backend of GLOO, MPI, and NCCL. Among these three, MPI backend since it can be used for both CPU and CUDA runnings, is the main backend we use for developement. Unfortunately installing the built version of PyTorch does not support MPI backend for distributed training and needed to be built from source with a version of MPI installed that supports CUDA as well. However, do not worry since we got you covered. We provide a docker file that can create an image with all dependencies needed for FedTorch. The Dockerfile can be found [here](docker/README.md), where you can edit based on your need to create your customized image. In addition, since building this docker image might take a lot of time, we provide different versions that we built before along with this repository in the [packages](https://github.com/orgs/MLOPTPSU/packages?repo_name=FedTorch) section. 

For instance, you can pull one of the images that is built with CUDA 10.2 and OpenMPI 4.0.1 with CUDA support and PyTorch 1.6.0, using the following command:
```cli
docker pull docker.pkg.github.com/mloptpsu/fedtorch/fedtorch:cuda10.2-mpi
```
The docker images can be used for cloud services such as [Azure Machine Learning API](https://azure.microsoft.com/en-us/services/machine-learning/) for running FedTorch. The instructions for running on cloud services will be added in near future.


## Get Started
Running different trainings with different settings is easy in FedTorch. We use config files in similar to those in `configs` folder. You can either use the predefined one and change some parameters inside them as instructed below, or to create from scratch by stacking different part bases. For different algorithms we will provide examples, so the relevant parameters can be set correctly. 

### Config File Structure
The configuration system is built on a hierarchical structure that promotes reuse and easy customization. At the core, the configuration is divided into several base files, each dedicated to a specific aspect of the training setup:

**Model Configuration**: Defines the architecture of the model to be trained.
**Dataset Configuration**: Specifies the dataset to be used, including paths, preprocessing, and other relevant details.
**Optimizer Configuration**: Contains settings for the optimizer, including learning rate, momentum, etc.
**Scheduler Configuration**: Outlines the learning rate scheduler to adjust the learning rate during training.
**Checkpoint Configuration**: Manages the saving and loading of model weights and training state.
**Device Configuration**: Determines the computing device (CPU/GPU) settings for the training process.
**Partitioner Configuration**: Configures the data partitioning strategy for federated learning.
**Training Configuration**: General training settings, such as epochs, batch size, etc.
**Federated Learning Configuration**: Specific settings for federated learning, like aggregation method and communication frequency.

Each configuration file can be referenced by others to build a complete and customized training setup. This modular approach facilitates easy adjustments and scaling of configurations to meet different training requirements. The configs specified in the `./configs/__base__` folder include different variations for each of these categories and their respective parameters.

### Example: FedAvg training

As an example, we can look at the FedAvg training, with example configuration on `./configs/fedavg_exps/fedavg_mnist_mlp_2class_local_step_10_centered.py`. The provided example demonstrates a federated learning setup using the Federated Averaging (FedAvg) algorithm. The configuration is composed of several base files, each tailored for a component of the federated learning system:

```python
_base_ = [
    '../_base_/model/mlp.py', # Model config
    '../_base_/dataset/mnist.py', # Data Config
    '../_base_/optimizer/sgd.py', # Optimizer Config
    '../_base_/scheduler/convex_decay.py', # Scheduler Config
    '../_base_/checkpoint.py', # Checkpoint Config
    '../_base_/device.py', # Device Config
    '../_base_/partitioner/federated_partitioner.py', # Partitioner Config
    '../_base_/training/federated.py', # Training Config
    '../_base_/federated/fedavg.py',  # Federated Learning Config
]
```
This config file will run a fedavg training (defined in `'../_base_/federated/fedavg.py'` file) on MNIST dataset (defined on `'../_base_/dataset/mnist.py'`), with MLP models (defined on `'../_base_/model/mlp.py'`) on 10 clients (defined on `'../_base_/device.py'` file) for 100 round of communication (defined on `'../_base_/training/federated.py'` file). The MNIST dataset is partitioned in a federated way by attributing 1 class per client (defined on `'../_base_/partitioner/federated_partitioner.py'` file). To run this training simply try this script:
```cli
python run_config.py ./configs/fedavg_exps/fedavg_mnist_mlp_2class_local_step_10_centered.py
```

### Customized Configs
Customization can happen in 2 ways:

#### Creating a customized config file
To create your own customized config, you can use the following format:

```python
__base__ = [...]

# Parameter updates using dictionaries
# For instance
federated = dict(
    sync_type='local_step',
)
```
In this format, you can first list the base files that this config is built upon. This could happen either by directly calling base files (inside the `__base__` folder) for each of the mentioned components, or by using another custom config file that already includes those files as its __base__ list. Due to hierarchical nature of the config, it can get those configs from the respective bases. Then, you can modify each parameter for any part of the training configs using a dictionary format to set their value. These setting the value inside this custom config file will overwrite their original value from __base__ files. For instance, in this example, for the `federated` component, we change the `sync_type` value from its original `epoch` to `local_step`.

#### Using `--options key=value` format
Another way to customize an already created config file is to change the parameters when running the script in cli. For instance, for the `./configs/fedavg_exps/fedavg_mnist_mlp_2class_local_step_10_centered.py` config file, we have already change the `sync_type` to `local_step` using the mentioned method in the last part. Now, if we want to change it back to `epoch` when running the code, we can do so like this:

```cli
python run_config.py ./configs/fedavg_exps/fedavg_mnist_mlp_2class_local_step_10_centered.py --options federated.sync_type='epoch'
```
Note that instead of the `dict` format inside the file, we use the inheritance format for the script `federated.sync_type`. Based on the nature of the parameter it could be multiple nested features. For instance we have `data.dataset.download` for setting if we are allowed to download the dataset or not.

### Examples


<details>

<summary>APFL</summary>

To run APFL algorithm a simple command will be:

```cli
python run_config.py ./configs/apfl_exps/apfl_cifar10_cnn_1class_local_step_20_centered.py --options training.local_step=20 device.type='cuda' device.num_clients=50 partitioner.num_class_per_client=2
```
</details>


<details>

<summary>PERM</summary>

To run a PERM training we can use the following command:

```cli
python run_config.py ./configs/perm_exps/permsingle_mnist_mlp_1class_local_step_10_centered.py --options device.type="cuda" device.num_clients=50
```

We changed the device to run the training on GPU and change the default num_clients to 50.

</details>


## References 
For this repository there are several different references used for each training procedure. If you use this repository in your research, please cite the following paper:
```ref
@article{haddadpour2020federated,
  title={Federated learning with compression: Unified analysis and sharp guarantees},
  author={Haddadpour, Farzin and Kamani, Mohammad Mahdi and Mokhtari, Aryan and Mahdavi, Mehrdad},
  journal={arXiv preprint arXiv:2007.01154},
  year={2020}
}
```
Our other papers developed using this repository should be cited using the following bibitems:
```ref
@inproceedings{haddadpour2019local,
  title={Local sgd with periodic averaging: Tighter analysis and adaptive synchronization},
  author={Haddadpour, Farzin and Kamani, Mohammad Mahdi and Mahdavi, Mehrdad and Cadambe, Viveck},
  booktitle={Advances in Neural Information Processing Systems},
  pages={11082--11094},
  year={2019}
}
@article{deng2020distributionally,
  title={Distributionally Robust Federated Averaging},
  author={Deng, Yuyang and Kamani, Mohammad Mahdi and Mahdavi, Mehrdad},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
@article{deng2020adaptive,
  title={Adaptive Personalized Federated Learning},
  author={Deng, Yuyang and Kamani, Mohammad Mahdi and Mahdavi, Mehrdad},
  journal={arXiv preprint arXiv:2003.13461},
  year={2020}
}
```

### Acknowledgement and Disclaimer
This repository is developed, mainly by [MM. Kamani](https://github.com/mmkamani7), based on our group's research on distributed and federated learning algorithms. We also developed the works of other groups' proposed methods using FedTorch for a better comparison. However, this repo is not the official code for those methods other than our group's. Some parts of the initial stages of this repository were based on a forked repo of Local SGD code from Tao Lin, which is not public now.
# FedTorch
