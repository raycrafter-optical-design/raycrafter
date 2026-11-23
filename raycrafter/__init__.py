from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("raycrafter")
except PackageNotFoundError:
    # package is not installed
    pass
