"""Module for testing ``ase_notebook.configuration``."""

from ase_notebook.configuration import ViewConfig


def test_view_config_init():
    """Test initialisation."""
    config = ViewConfig()
    assert config.axes_length == 15
