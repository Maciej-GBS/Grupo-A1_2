"""
Vector implementation.\n
Optimized to work with maze solving intelligent system.
"""

class SortedVector:
    """SortedVector()\n
    Store elements in a sorted list"""
    def __init__(self):
        self.vector = []

    def __len__(self):
        # TODO: aqui tenemos que devolver vector length
        return 0

    def __getitem__(self, i):
        # con esto se puede acceder elementos como SortedVector_instance[index de elemento]
        # ya esta implementado
        return self.get(i)

    def __iter__(self):
        # con esto se puede hacer loops, por ejemplo: for x in SortedVector_inst: print(x)
        # ya esta implementado
        return iter(self.vector)

    def get(self, i):
        "Return data object under given index"
        # aqui devolvemos el valor de index specificado en una manera controlada
        # por ejemplo:
        return self.vector[i]

    def push(self, elem):
        "Insert element in ascending order"

    def pop(self):
        "Remove the first element and return its value"
        return None
