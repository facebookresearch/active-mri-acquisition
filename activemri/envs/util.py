import importlib


def import_object_from_str(classname: str):
    the_module = ".".join(classname.split(".")[:-1])
    the_object = classname.split(".")[-1]
    module = importlib.import_module(the_module)
    return getattr(module, the_object)
