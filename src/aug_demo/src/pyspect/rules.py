from abc import ABC, abstractmethod

from .spect import Spect

class Rule(ABC): 

    name: str
    approximation: str
    parameters: dict

    children: tuple

    def __init__(self, *children, name='', **kwargs):
        self.name = name 
        self.children = children
        self.parameters = kwargs

        self._apply_approximation

    def set_all(self, **parameters):
        self.parameters.update(parameters)
        for child in self.children:
            child.set_all(**parameters)

    def __repr__(self):
        op = type(self).__name__
        body = ',\n'.join(map(repr, self.children))
        sep = '\n' + ' ' * (len(op)+1)
        body = sep.join(body.splitlines())
        return f'{op}({body})'

    def __contains__(self, item):
        for child in self.children:
            if item in child:
                return True
        return False

    @abstractmethod
    def __call__(self, g: dict, r: dict) -> Spect:
        raise NotImplementedError()

    def _apply_approximation(self): 
        """
        1. If default approximation is 'exact', the first child decide type of approximation on parent
        2. If parent is under, no child can be over
        3. If parent is over, no child can be under
        """
        for child in self.children:

            if self.approximation == 'under':
                assert child.approximation != 'over' 
            elif self.approximation == 'over':
                assert child.approximation != 'under' 

            self.approximation = child.approximation

    def iter_props(self):
        for c in self.children:
            yield from c.iter_props()

class Proposition(Rule):

    def __init__(self, name, approximation='exact'): 
        self.approximation = approximation
        super().__init__(name=name)

    def __repr__(self):
        op = type(self).__name__
        name = self.name
        return f'{op}({name})'

    def __contains__(self, item):
        return item == self.name

    def __call__(self, g, r):

        s = g[self.name]

        if r is not None:
            cls_name = type(self).__name__
            r['...' + cls_name] = s

        return s

    def iter_props(self):
        yield self

class Not(Rule):

    approximation = 'exact'

    def __call__(self, g, r):
        return 

    def _apply_approximation(self):

        for child in self.children:

            if child.approximation == 'under':
                self.approximation = 'over'
            elif child.approximation == 'over':
                self.approximation = 'under'

    def __call__(self, g, r):
        for child in self.children:
            return child(g, r).complement()

class And(Rule):

    approximation = 'exact'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self.children) > 1, 'Must have at least two children'

    def __call__(self, g, r):

        s = self.children[0](g, r)

        for child in self.children[1:]:
            s = s.intersect(child(g, r))

        if r is not None:
            cls_name = type(self).__name__
            r['...' + cls_name] = s

        return s

class Or(Rule): 

    approximation = 'exact'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self.children) > 1, 'Must hahve at least two children'

    def __call__(self, g, r):

        s = self.children[0](g, r)

        for child in self.children[1:]:
            s = s.union(child(g, r))

        return s


class Eventually(Rule): 

    approximation = 'under'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self.children) == 1, 'Can only have one child'

    def __call__(self, g, r):

        child = self.children[0]

        s = child(g, r).brs(**self.parameters)

        if r is not None:
            cls_name = type(self).__name__
            r['...' + cls_name] = s

        return s

class Always(Rule):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self.children) == 1, 'Can only have one child'

class Until(Rule):
    
    approximation = 'under' 

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert len(self.children) == 2, 'Can only have one child'

    def __call__(self, g, r):

        precondition, postcondition = self.children

"""

def solve(rule, save_intermediary=False, **props):

    g: dict[str, Set] = props

    r = {} if save_intermediary else None

    s: Set = rule(g, r)

    return s, r

if __name__ == '__main__':

    rule1 = And(Always(Eventually(Proposition('p1'))),
                Always(Proposition('p2')),
                name='phi1')

    s, r = solve(rule1)

    if s.empty():
        print('infeasible!')
    else:
        print('feasible!')

"""
