# determine the package version
try:
    # try reading version of the automatically generated module
    from ._version import __version__
except ImportError:
    # determine version automatically from CVS information
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("pyphasediagram")
    except PackageNotFoundError:
        # package is not installed, so we cannot determine the version
        __version__ = "unknown"
    del PackageNotFoundError, version  # clean name space

from pyphasediagram.diagram import PhaseDiagram

# from pyphasediagram.stepper import TernaryStepper
# from pyphasediagram.stepper import BaseStepper
