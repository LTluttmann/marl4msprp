import importlib
import pkgutil

def import_submodules(package):
    package = importlib.import_module(package)
    for _, module_name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
        importlib.import_module(module_name)

# Automatically import all modules in the current package
import_submodules(__name__)