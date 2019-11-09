"""Module for visualizing atomic configurations."""
try:
    from .backend.svg import concatenate_svgs, svg_to_pdf  # noqa: F401
    from .color import Color  # noqa: F401
    from .configuration import ViewConfig  # noqa: F401
    from .viewer import AseView  # noqa: F401
except ImportError:
    # this is required by setup.py
    # before dependencies are installed
    pass

__version__ = "0.1.0"
