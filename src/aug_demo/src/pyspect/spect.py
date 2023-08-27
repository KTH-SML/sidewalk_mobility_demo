from abc import ABC, abstractmethod
from typing_extensions import Self

class Spect(ABC): 

    approximation: str

    def __init__(self, approximation: str):
        self.approximation = approximation
        assert self.approximation in ('over', 'exact', 'under'), 'Invalid approximation type'

    def _overloads(self, name: str):
        return getattr(type(self), name) is getattr(Spect, name)

    def _overloaded_methods(self):
        return list(filter(self._overloads, [
            'empty',
            'complement',
            'union',
            'intersect',
            'brs',
            'frs',
            'rci',
            'membership',
            'control_set',
        ]))

    @abstractmethod
    def empty(self): pass 

    def complement(self): pass

    def union(self, other: Self): pass

    def intersect(self, other: Self): pass

    def brs(self): pass

    def frs(self): pass

    def rci(self): pass

    def membership(self, point): pass

    def control_set(self, point): pass

