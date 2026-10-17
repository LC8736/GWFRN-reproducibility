"""Load only the installed GNNWR's pure-Python network definitions.

The installed package __init__ eagerly imports datasets, full training models,
and sklearn. This experiment needs networks.py only, whose dependencies are
math and PyTorch. No binary security setting or blocked DLL is changed/loaded.
"""
from importlib import util, machinery
from pathlib import Path
import sys
import types


def load_networks_only():
    if "gnnwr.networks" in sys.modules:
        return
    spec = machinery.PathFinder.find_spec("gnnwr", sys.path)
    if spec is None or spec.origin is None:
        raise ImportError("The official gnnwr distribution must be installed")
    folder = Path(spec.origin).parent
    network_file = folder / "networks.py"
    if "gnnwr" not in sys.modules:
        package = types.ModuleType("gnnwr")
        package.__path__ = [str(folder)]
        package.__package__ = "gnnwr"
        package.__spec__ = machinery.ModuleSpec("gnnwr", loader=None, is_package=True)
        sys.modules["gnnwr"] = package
    sub_spec = util.spec_from_file_location("gnnwr.networks", network_file)
    module = util.module_from_spec(sub_spec)
    sys.modules["gnnwr.networks"] = module
    sub_spec.loader.exec_module(module)
    sys.modules["gnnwr"].networks = module

