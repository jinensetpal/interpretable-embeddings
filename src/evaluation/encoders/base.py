#!/usr/bin/env python3

from abc import ABC, abstractmethod


class BaseEncoder(ABC):

    @abstractmethod
    def encode(self, batch):
        pass
