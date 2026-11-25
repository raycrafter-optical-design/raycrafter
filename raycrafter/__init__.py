"""RayCrafter: lens-level optical design with off-the-shelf lenses.

To script it from Python, use ``Bench`` (see ``raycrafter.api``):

>>> from raycrafter import Bench
>>> bench = Bench()
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("raycrafter")
except PackageNotFoundError:
    # package is not installed
    pass


def __getattr__(name: str):
    """Import ``Bench`` only when it is asked for.

    ``import raycrafter`` happens whenever any part of the package is used (the GUI, the MCP
    server, the lens catalogue on its own), so it must not load Optiland and matplotlib.
    """
    if name == "Bench":
        from raycrafter.api import Bench

        return Bench
    raise AttributeError(f"module 'raycrafter' has no attribute {name!r}")
