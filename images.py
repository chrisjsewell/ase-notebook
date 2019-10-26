"""Module for manipulating images."""


def pil_hconcat(images, width=700, height=700, gap=0):
    """Concatenate multiple images horizontally.

    Parameters
    ----------
    images : list[PIL.Image]
    width : int or list[int]
        maximum width of final image, or of individual images
    height : int or list[int]
        maximum height of final image, or of individual images
    gap : int
        size of space between images

    Returns
    -------
    image : PIL.Image

    """
    from PIL import Image

    if not isinstance(width, list):
        widths = [width for _ in images]
    else:
        widths = width[:]
        width = sum(widths) + gap * (len(images) - 1)
    if not isinstance(height, list):
        heights = [height for _ in images]
    else:
        heights = height[:]
        height = sum(heights)
    for im, w, h in zip(images, widths, heights):
        im.thumbnail((w, h), Image.ANTIALIAS)
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths) + gap * (len(images) - 1)
    max_height = max(heights)
    new_im = Image.new("RGBA", (total_width, max_height))
    x_offset = 0
    for im in images:
        im = im.convert("RGBA")
        new_im.paste(im, (x_offset, 0), mask=im)
        x_offset += im.size[0] + gap
    new_im.thumbnail((width, height), Image.ANTIALIAS)
    return new_im
