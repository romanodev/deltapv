import yaml
import glob
import os

MATERIAL_FILES = glob.glob(
    os.path.join(os.path.dirname(__file__), "resources/*.yaml"))


class Material(object):
    def __init__(self):
        self.materials = {}
        self.required = {
            "eps": 10.,
            "Chi": 5.,
            "Eg": 1.,
            "Nc": 1e18,
            "Nv": 1e18,
            "mn": 100.,
            "mp": 100.
        }
        self.optional = {"Et", "tn", "tp", "Br", "Cn", "Cp", "A"}
        for file in MATERIAL_FILES:
            try:
                with open(file, "r") as f:
                    mat = yaml.full_load(f)
                    self.materials[mat["name"]] = mat["properties"]
            except:
                pass

    def __getitem__(self, key):
        assert key in self.materials, "Material not in database!"
        mat = {}
        for prop in self.required:
            value = self.materials[key][prop]
            mat[prop] = value if value is not None else self.required[key]
        for prop in self.optional:
            value = self.materials[key][prop]
            if value is not None:
                mat[prop] = value
        return mat
