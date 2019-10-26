"""Tests for ``aiida_2d.visualize.images``."""

from aiida_2d.visualize import images


def test_pil_hconcat(get_test_filepath):
    """Test concatenating two PIL images."""
    from PIL import Image

    image1 = Image.open(get_test_filepath("pyrite.eps")).convert("RGBA")
    image2 = Image.open(get_test_filepath("pyrite.png")).convert("RGBA")
    concated = images.pil_hconcat([image1, image2])
    assert isinstance(concated, Image.Image), concated
