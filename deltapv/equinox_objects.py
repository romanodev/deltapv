def eqx_dataclass(data_clz):
    # Uses equinox to create a new class with the same fields as the original
    # and getter methods for pytrees

    class clz(data_clz):

        items = data_clz.__dataclass_fields__.items()
        
        def iterate_clz(self, x):
            # iterates the class and returns a tuple of every field
            return tuple(getattr(x, name) for name in self.items)
        
        def clz_from_iterable(self, data):
            # creates a new class from a tuple of fields
            kwargs = dict(zip(self.items, data))
            return data_clz(**kwargs)

    return clz