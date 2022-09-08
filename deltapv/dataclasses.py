'''
DEPRECATED
'''


import dataclasses
import jax


def dataclass(clz):

    data_clz = dataclasses.dataclass(frozen=True)(clz)
    meta_fields = []
    data_fields = []
    for name, field_info in data_clz.__dataclass_fields__.items():
        is_static = field_info.metadata.get('static', False)
        if is_static:
            meta_fields.append(name)
        else:
            data_fields.append(name)

    def iterate_clz(x):
        meta = tuple(getattr(x, name) for name in meta_fields)
        data = tuple(getattr(x, name) for name in data_fields)
        return data, meta

    def clz_from_iterable(meta, data):
        meta_args = tuple(zip(meta_fields, meta))
        data_args = tuple(zip(data_fields, data))
        kwargs = dict(meta_args + data_args)
        return data_clz(**kwargs)

    jax.tree_util.register_pytree_node(data_clz, iterate_clz,
                                       clz_from_iterable)

    return data_clz


def static_field():
    return dataclasses.field(metadata={'static': True})


replace = dataclasses.replace
asdict = dataclasses.asdict
astuple = dataclasses.astuple
