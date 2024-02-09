# -*- coding: utf-8 -*-

from itertools import groupby
from functools import reduce

from abc import abstractmethod, ABCMeta
import numpy as np
import torch


class UndirectedGraph(metaclass=ABCMeta):

    @property
    @abstractmethod
    def n_nodes(self):
        pass

    @property
    @abstractmethod
    def world(self):
        pass

    @property
    @abstractmethod
    def rank(self):
        pass

    @property
    @abstractmethod
    def ranks(self):
        pass

    @property
    @abstractmethod
    def ranks_with_blocks(self):
        pass

    @property
    @abstractmethod
    def blocks_with_ranks(self):
        pass

    @property
    @abstractmethod
    def device(self):
        pass

    @property
    @abstractmethod
    def device_type(self):
        pass

    @abstractmethod
    def get_neighborhood(self, node_id):
        pass


class FCGraph(UndirectedGraph):
    def __init__(self, rank, blocks, device_type, world=None):
        self._rank = rank
        self._blocks = [int(l) for l in blocks.split(',')]
        self._device_type = device_type
        self._world = world
        self.debug = True
        self.torch_device = torch.device(self._device_type)

    @property
    def n_nodes(self):
        return sum(self._blocks)

    @property
    def world(self):
        return reduce(
            lambda a, b: a + b, [list(range(b)) for b in self._blocks]) \
            if self._world is None \
            else [int(l) for l in self._world.split(',')]

    @property
    def rank(self):
        return self._rank

    @property
    def ranks(self):
        return list(range(self.n_nodes))

    @property
    def ranks_with_blocks(self):
        return self._assign_ranks_with_blocks()

    @property
    def blocks_with_ranks(self):
        return dict(
            [(k, [l[0] for l in g])
             for k, g in groupby(self.ranks_with_blocks.items(),
                                 lambda x: x[1])]
            )

    @property
    def device(self):
        return self.world[self.rank]

    @property
    def device_type(self):
        return self._device_type

    def get_neighborhood(self):
        """it will return a list of ranks that are connected with this node."""
        return self.blocks_with_ranks[self.ranks_with_blocks[self.rank]]

    def _assign_ranks_with_blocks(self):
        blocks = []
        for block_ind, block_size in enumerate(self._blocks):
            blocks += [block_ind] * block_size
        return dict(list(zip(self.ranks, blocks)))

    def _get_device_type(self):
        pass
