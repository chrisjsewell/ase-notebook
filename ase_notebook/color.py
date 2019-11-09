# -*- coding: utf-8 -*-
"""Color Library.

Adapted from: https://github.com/vaab/colour/blob/master/colour.py
vaab/colour is licensed under the BSD 2-Clause "Simplified" License

.. :doctest:

This module defines several color formats that can be converted to one or
another.

Formats
-------

HSL:
    3-ple of Hue, Saturation, Lightness all between 0.0 and 1.0

RGB:
    3-ple of Red, Green, Blue all between 0.0 and 1.0

HEX:
    string object beginning with '#' and with red, green, blue value.
    This format accept color in 3 or 6 value ex: '#fff' or '#ffffff'

WEB:
    string object that defaults to HEX representation or human if possible

Usage
-----

Several function exists to convert from one format to another. But all
function are not written. So the best way is to use the object Color.

Please see the documentation of this object for more information.

.. note:: Some constants are defined for convenience in HSL, RGB, HEX

"""
import re

# Soften inequalities and some rounding issue based on float
FLOAT_ERROR = 0.0000005


RGB_TO_COLOR_NAMES = {
    (0, 0, 0): ["Black"],
    (0, 0, 128): ["Navy", "NavyBlue"],
    (0, 0, 139): ["DarkBlue"],
    (0, 0, 205): ["MediumBlue"],
    (0, 0, 255): ["Blue"],
    (0, 100, 0): ["DarkGreen"],
    (0, 128, 0): ["Green"],
    (0, 139, 139): ["DarkCyan"],
    (0, 191, 255): ["DeepSkyBlue"],
    (0, 206, 209): ["DarkTurquoise"],
    (0, 250, 154): ["MediumSpringGreen"],
    (0, 255, 0): ["Lime"],
    (0, 255, 127): ["SpringGreen"],
    (0, 255, 255): ["Cyan", "Aqua"],
    (25, 25, 112): ["MidnightBlue"],
    (30, 144, 255): ["DodgerBlue"],
    (32, 178, 170): ["LightSeaGreen"],
    (34, 139, 34): ["ForestGreen"],
    (46, 139, 87): ["SeaGreen"],
    (47, 79, 79): ["DarkSlateGray", "DarkSlateGrey"],
    (50, 205, 50): ["LimeGreen"],
    (60, 179, 113): ["MediumSeaGreen"],
    (64, 224, 208): ["Turquoise"],
    (65, 105, 225): ["RoyalBlue"],
    (70, 130, 180): ["SteelBlue"],
    (72, 61, 139): ["DarkSlateBlue"],
    (72, 209, 204): ["MediumTurquoise"],
    (75, 0, 130): ["Indigo"],
    (85, 107, 47): ["DarkOliveGreen"],
    (95, 158, 160): ["CadetBlue"],
    (100, 149, 237): ["CornflowerBlue"],
    (102, 205, 170): ["MediumAquamarine"],
    (105, 105, 105): ["DimGray", "DimGrey"],
    (106, 90, 205): ["SlateBlue"],
    (107, 142, 35): ["OliveDrab"],
    (112, 128, 144): ["SlateGray", "SlateGrey"],
    (119, 136, 153): ["LightSlateGray", "LightSlateGrey"],
    (123, 104, 238): ["MediumSlateBlue"],
    (124, 252, 0): ["LawnGreen"],
    (127, 255, 0): ["Chartreuse"],
    (127, 255, 212): ["Aquamarine"],
    (128, 0, 0): ["Maroon"],
    (128, 0, 128): ["Purple"],
    (128, 128, 0): ["Olive"],
    (128, 128, 128): ["Gray", "Grey"],
    (132, 112, 255): ["LightSlateBlue"],
    (135, 206, 235): ["SkyBlue"],
    (135, 206, 250): ["LightSkyBlue"],
    (138, 43, 226): ["BlueViolet"],
    (139, 0, 0): ["DarkRed"],
    (139, 0, 139): ["DarkMagenta"],
    (139, 69, 19): ["SaddleBrown"],
    (143, 188, 143): ["DarkSeaGreen"],
    (144, 238, 144): ["LightGreen"],
    (147, 112, 219): ["MediumPurple"],
    (148, 0, 211): ["DarkViolet"],
    (152, 251, 152): ["PaleGreen"],
    (153, 50, 204): ["DarkOrchid"],
    (154, 205, 50): ["YellowGreen"],
    (160, 82, 45): ["Sienna"],
    (165, 42, 42): ["Brown"],
    (169, 169, 169): ["DarkGray", "DarkGrey"],
    (173, 216, 230): ["LightBlue"],
    (173, 255, 47): ["GreenYellow"],
    (175, 238, 238): ["PaleTurquoise"],
    (176, 196, 222): ["LightSteelBlue"],
    (176, 224, 230): ["PowderBlue"],
    (178, 34, 34): ["Firebrick"],
    (184, 134, 11): ["DarkGoldenrod"],
    (186, 85, 211): ["MediumOrchid"],
    (188, 143, 143): ["RosyBrown"],
    (189, 183, 107): ["DarkKhaki"],
    (192, 192, 192): ["Silver"],
    (199, 21, 133): ["MediumVioletRed"],
    (205, 92, 92): ["IndianRed"],
    (205, 133, 63): ["Peru"],
    (208, 32, 144): ["VioletRed"],
    (210, 105, 30): ["Chocolate"],
    (210, 180, 140): ["Tan"],
    (211, 211, 211): ["LightGray", "LightGrey"],
    (216, 191, 216): ["Thistle"],
    (218, 112, 214): ["Orchid"],
    (218, 165, 32): ["Goldenrod"],
    (219, 112, 147): ["PaleVioletRed"],
    (220, 20, 60): ["Crimson"],
    (220, 220, 220): ["Gainsboro"],
    (221, 160, 221): ["Plum"],
    (222, 184, 135): ["Burlywood"],
    (224, 255, 255): ["LightCyan"],
    (230, 230, 250): ["Lavender"],
    (233, 150, 122): ["DarkSalmon"],
    (238, 130, 238): ["Violet"],
    (238, 221, 130): ["LightGoldenrod"],
    (238, 232, 170): ["PaleGoldenrod"],
    (240, 128, 128): ["LightCoral"],
    (240, 230, 140): ["Khaki"],
    (240, 248, 255): ["AliceBlue"],
    (240, 255, 240): ["Honeydew"],
    (240, 255, 255): ["Azure"],
    (244, 164, 96): ["SandyBrown"],
    (245, 222, 179): ["Wheat"],
    (245, 245, 220): ["Beige"],
    (245, 245, 245): ["WhiteSmoke"],
    (245, 255, 250): ["MintCream"],
    (248, 248, 255): ["GhostWhite"],
    (250, 128, 114): ["Salmon"],
    (250, 235, 215): ["AntiqueWhite"],
    (250, 240, 230): ["Linen"],
    (250, 250, 210): ["LightGoldenrodYellow"],
    (253, 245, 230): ["OldLace"],
    (255, 0, 0): ["Red"],
    (255, 0, 255): ["Magenta", "Fuchsia"],
    (255, 20, 147): ["DeepPink"],
    (255, 69, 0): ["OrangeRed"],
    (255, 99, 71): ["Tomato"],
    (255, 105, 180): ["HotPink"],
    (255, 127, 80): ["Coral"],
    (255, 140, 0): ["DarkOrange"],
    (255, 160, 122): ["LightSalmon"],
    (255, 165, 0): ["Orange"],
    (255, 182, 193): ["LightPink"],
    (255, 192, 203): ["Pink"],
    (255, 215, 0): ["Gold"],
    (255, 218, 185): ["PeachPuff"],
    (255, 222, 173): ["NavajoWhite"],
    (255, 228, 181): ["Moccasin"],
    (255, 228, 196): ["Bisque"],
    (255, 228, 225): ["MistyRose"],
    (255, 235, 205): ["BlanchedAlmond"],
    (255, 239, 213): ["PapayaWhip"],
    (255, 240, 245): ["LavenderBlush"],
    (255, 245, 238): ["Seashell"],
    (255, 248, 220): ["Cornsilk"],
    (255, 250, 205): ["LemonChiffon"],
    (255, 250, 240): ["FloralWhite"],
    (255, 250, 250): ["Snow"],
    (255, 255, 0): ["Yellow"],
    (255, 255, 224): ["LightYellow"],
    (255, 255, 240): ["Ivory"],
    (255, 255, 255): ["White"],
}

# Building inverse relation
COLOR_NAME_TO_RGB = {
    name.lower(): rgb for rgb, names in RGB_TO_COLOR_NAMES.items() for name in names
}


LONG_HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")
SHORT_HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{3}$")


class ContainerHSL:
    """HSL colors container."""

    def __getattr__(self, value):
        """Get attribute."""
        label = value.lower()
        if label in COLOR_NAME_TO_RGB:
            return rgb2hsl(tuple(v / 255.0 for v in COLOR_NAME_TO_RGB[label]))
        raise AttributeError(
            "%s instance has no attribute %r" % (self.__class__, value)
        )


HSL = ContainerHSL()


class ContainerRGB:
    """RGB colors container.

    Provides a quick color access.

    >>> from ase_notebook.color import RGB

    >>> RGB.WHITE
    (1.0, 1.0, 1.0)
    >>> RGB.BLUE
    (0.0, 0.0, 1.0)

    >>> RGB.DONOTEXISTS  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    AttributeError: ... has no attribute 'DONOTEXISTS'

    """

    def __getattr__(self, value):
        """Get attribute."""
        return hsl2rgb(getattr(HSL, value))


class ContainerHEX:
    """RGB colors container.

    Provides a quick color access.

    >>> from ase_notebook.color import HEX

    >>> HEX.WHITE
    '#fff'
    >>> HEX.BLUE
    '#00f'

    >>> HEX.DONOTEXISTS  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    AttributeError: ... has no attribute 'DONOTEXISTS'

    """

    def __getattr__(self, value):
        """Get attribute."""
        return rgb2hex(getattr(RGB, value))


RGB = ContainerRGB()
HEX = ContainerHEX()


#
# Conversion function
#


def hsl2rgb(hsl):
    """Convert HSL representation towards RGB.

    :param h: Hue, position around the chromatic circle (h=1 equiv h=0)
    :param s: Saturation, color saturation (0=full gray, 1=full color)
    :param l: Ligthness, Overhaul lightness (0=full black, 1=full white)
    :rtype: 3-uple for RGB values in float between 0 and 1

    Hue, Saturation, Range from Lightness is a float between 0 and 1

    Note that Hue can be set to any value but as it is a rotation
    around the chromatic circle, any value above 1 or below 0 can
    be expressed by a value between 0 and 1 (Note that h=0 is equiv
    to h=1).

    This algorithm came from:
    http://www.easyrgb.com/index.php?X=MATH&H=19#text19

    Here are some quick notion of HSL to RGB conversion:

    >>> from ase_notebook.color import hsl2rgb

    With a lightness put at 0, RGB is always rgbblack

    >>> hsl2rgb((0.0, 0.0, 0.0))
    (0.0, 0.0, 0.0)
    >>> hsl2rgb((0.5, 0.0, 0.0))
    (0.0, 0.0, 0.0)
    >>> hsl2rgb((0.5, 0.5, 0.0))
    (0.0, 0.0, 0.0)

    Same for lightness put at 1, RGB is always rgbwhite

    >>> hsl2rgb((0.0, 0.0, 1.0))
    (1.0, 1.0, 1.0)
    >>> hsl2rgb((0.5, 0.0, 1.0))
    (1.0, 1.0, 1.0)
    >>> hsl2rgb((0.5, 0.5, 1.0))
    (1.0, 1.0, 1.0)

    With saturation put at 0, the RGB should be equal to Lightness:

    >>> hsl2rgb((0.0, 0.0, 0.25))
    (0.25, 0.25, 0.25)
    >>> hsl2rgb((0.5, 0.0, 0.5))
    (0.5, 0.5, 0.5)
    >>> hsl2rgb((0.5, 0.0, 0.75))
    (0.75, 0.75, 0.75)

    With saturation put at 1, and lightness put to 0.5, we can find
    normal full red, green, blue colors:

    >>> hsl2rgb((0 , 1.0, 0.5))
    (1.0, 0.0, 0.0)
    >>> hsl2rgb((1 , 1.0, 0.5))
    (1.0, 0.0, 0.0)
    >>> hsl2rgb((1.0/3 , 1.0, 0.5))
    (0.0, 1.0, 0.0)
    >>> hsl2rgb((2.0/3 , 1.0, 0.5))
    (0.0, 0.0, 1.0)

    Of course:
    >>> hsl2rgb((0.0, 2.0, 0.5))  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: Saturation must be between 0 and 1.

    And:
    >>> hsl2rgb((0.0, 0.0, 1.5))  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: Lightness must be between 0 and 1.

    """
    h, s, l = [float(v) for v in hsl]  # noqa E741

    if not (0.0 - FLOAT_ERROR <= s <= 1.0 + FLOAT_ERROR):
        raise ValueError("Saturation must be between 0 and 1.")
    if not (0.0 - FLOAT_ERROR <= l <= 1.0 + FLOAT_ERROR):  # noqa E741
        raise ValueError("Lightness must be between 0 and 1.")

    if s == 0:
        return l, l, l

    if l < 0.5:
        v2 = l * (1.0 + s)
    else:
        v2 = (l + s) - (s * l)

    v1 = 2.0 * l - v2

    r = _hue2rgb(v1, v2, h + (1.0 / 3))
    g = _hue2rgb(v1, v2, h)
    b = _hue2rgb(v1, v2, h - (1.0 / 3))

    return r, g, b


def rgb2hsl(rgb):
    """Convert RGB representation towards HSL.

    :param r: Red amount (float between 0 and 1)
    :param g: Green amount (float between 0 and 1)
    :param b: Blue amount (float between 0 and 1)
    :rtype: 3-uple for HSL values in float between 0 and 1

    This algorithm came from:
    http://www.easyrgb.com/index.php?X=MATH&H=19#text19

    Here are some quick notion of RGB to HSL conversion:

    >>> from ase_notebook.color import rgb2hsl

    Note that if red amount is equal to green and blue, then you
    should have a gray value (from black to white).


    >>> rgb2hsl((1.0, 1.0, 1.0))  # doctest: +ELLIPSIS
    (..., 0.0, 1.0)
    >>> rgb2hsl((0.5, 0.5, 0.5))  # doctest: +ELLIPSIS
    (..., 0.0, 0.5)
    >>> rgb2hsl((0.0, 0.0, 0.0))  # doctest: +ELLIPSIS
    (..., 0.0, 0.0)

    If only one color is different from the others, it defines the
    direct Hue:

    >>> rgb2hsl((0.5, 0.5, 1.0))  # doctest: +ELLIPSIS
    (0.66..., 1.0, 0.75)
    >>> rgb2hsl((0.2, 0.1, 0.1))  # doctest: +ELLIPSIS
    (0.0, 0.33..., 0.15...)

    Having only one value set, you can check that:

    >>> rgb2hsl((1.0, 0.0, 0.0))
    (0.0, 1.0, 0.5)
    >>> rgb2hsl((0.0, 1.0, 0.0))  # doctest: +ELLIPSIS
    (0.33..., 1.0, 0.5)
    >>> rgb2hsl((0.0, 0.0, 1.0))  # doctest: +ELLIPSIS
    (0.66..., 1.0, 0.5)

    Regression check upon very close values in every component of
    red, green and blue:

    >>> rgb2hsl((0.9999999999999999, 1.0, 0.9999999999999994))
    (0.0, 0.0, 0.999...)

    Of course:

    >>> rgb2hsl((0.0, 2.0, 0.5))  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: Green must be between 0 and 1. You provided 2.0.

    And:
    >>> rgb2hsl((0.0, 0.0, 1.5))  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: Blue must be between 0 and 1. You provided 1.5.

    """
    r, g, b = [float(v) for v in rgb]

    for name, v in {"Red": r, "Green": g, "Blue": b}.items():
        if not (0 - FLOAT_ERROR <= v <= 1 + FLOAT_ERROR):
            raise ValueError("%s must be between 0 and 1. You provided %r." % (name, v))

    vmin = min(r, g, b)  # Min. value of RGB
    vmax = max(r, g, b)  # Max. value of RGB
    diff = vmax - vmin  # Delta RGB value

    vsum = vmin + vmax

    l = vsum / 2  # noqa E741

    if diff < FLOAT_ERROR:  # This is a gray, no chroma...
        return (0.0, 0.0, l)

    #
    # Chromatic data...
    #

    # Saturation
    if l < 0.5:
        s = diff / vsum
    else:
        s = diff / (2.0 - vsum)

    dr = (((vmax - r) / 6) + (diff / 2)) / diff
    dg = (((vmax - g) / 6) + (diff / 2)) / diff
    db = (((vmax - b) / 6) + (diff / 2)) / diff

    if r == vmax:
        h = db - dg
    elif g == vmax:
        h = (1.0 / 3) + dr - db
    elif b == vmax:
        h = (2.0 / 3) + dg - dr

    if h < 0:
        h += 1
    if h > 1:
        h -= 1

    return (h, s, l)


def _hue2rgb(v1, v2, v_h):
    """Private helper function (Do not call directly).

    :param vH: rotation around the chromatic circle (between 0..1)

    """
    while v_h < 0:
        v_h += 1
    while v_h > 1:
        v_h -= 1

    if 6 * v_h < 1:
        return v1 + (v2 - v1) * 6 * v_h
    if 2 * v_h < 1:
        return v2
    if 3 * v_h < 2:
        return v1 + (v2 - v1) * ((2.0 / 3) - v_h) * 6

    return v1


def rgb2hex(rgb, force_long=False):
    """Transform RGB tuple to hex RGB representation.

    :param rgb: RGB 3-uple of float between 0 and 1
    :rtype: 3 hex char or 6 hex char string representation

    Usage
    -----

    >>> from ase_notebook.color import rgb2hex

    >>> rgb2hex((0.0,1.0,0.0))
    '#0f0'

    Rounding try to be as natural as possible:

    >>> rgb2hex((0.0,0.999999,1.0))
    '#0ff'

    And if not possible, the 6 hex char representation is used:

    >>> rgb2hex((0.23,1.0,1.0))
    '#3bffff'

    >>> rgb2hex((0.0,0.999999,1.0), force_long=True)
    '#00ffff'

    """
    hx = "".join(["%02x" % int(c * 255 + 0.5 - FLOAT_ERROR) for c in rgb])

    if not force_long and hx[0::2] == hx[1::2]:
        hx = "".join(hx[0::2])

    return "#%s" % hx


def hex2rgb(str_rgb):
    """Transform hex RGB representation to RGB tuple.

    :param str_rgb: 3 hex char or 6 hex char string representation
    :rtype: RGB 3-uple of float between 0 and 1

    >>> from ase_notebook.color import hex2rgb

    >>> hex2rgb('#00ff00')
    (0.0, 1.0, 0.0)

    >>> hex2rgb('#0f0')
    (0.0, 1.0, 0.0)

    >>> hex2rgb('#aaa')  # doctest: +ELLIPSIS
    (0.66..., 0.66..., 0.66...)

    >>> hex2rgb('#aa')  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: Invalid value '#aa' provided for rgb color.

    """
    try:
        rgb = str_rgb[1:]

        if len(rgb) == 6:
            r, g, b = rgb[0:2], rgb[2:4], rgb[4:6]
        elif len(rgb) == 3:
            r, g, b = rgb[0] * 2, rgb[1] * 2, rgb[2] * 2
        else:
            raise ValueError()
    except Exception:
        raise ValueError("Invalid value %r provided for rgb color." % str_rgb)

    return tuple(float(int(v, 16)) / 255 for v in (r, g, b))


def hex2web(hex_color):
    """Convert HEX representation to WEB.

    :param rgb: 3 hex char or 6 hex char string representation
    :rtype: web string representation (human readable if possible)

    WEB representation uses X11 rgb.txt to define conversion
    between RGB and english color names.

    Usage
    =====

    >>> from ase_notebook.color import hex2web

    >>> hex2web('#ff0000')
    'red'

    >>> hex2web('#aaaaaa')
    '#aaa'

    >>> hex2web('#abc')
    '#abc'

    >>> hex2web('#acacac')
    '#acacac'

    """
    dec_rgb = tuple(int(v * 255) for v in hex2rgb(hex_color))
    if dec_rgb in RGB_TO_COLOR_NAMES:
        # take the first one
        color_name = RGB_TO_COLOR_NAMES[dec_rgb][0]
        # Enforce full lowercase for single worded color name.
        return (
            color_name
            if len(re.sub(r"[^A-Z]", "", color_name)) > 1
            else color_name.lower()
        )

    # Hex format is verified by hex2rgb function. And should be 3 or 6 digit
    if len(hex_color) == 7:
        if (
            hex_color[1] == hex_color[2]
            and hex_color[3] == hex_color[4]
            and hex_color[5] == hex_color[6]
        ):
            return "#" + hex_color[1] + hex_color[3] + hex_color[5]
    return hex_color


def web2hex(web, force_long=False):
    """Convert WEB representation to HEX.

    :param rgb: web string representation (human readable if possible)
    :rtype: 3 hex char or 6 hex char string representation

    WEB representation uses X11 rgb.txt to define conversion
    between RGB and english color names.

    Usage
    =====

    >>> from ase_notebook.color import web2hex

    >>> web2hex('red')
    '#f00'

    >>> web2hex('#aaa')
    '#aaa'

    >>> web2hex('#foo')  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    AttributeError: '#foo' is not in web format. Need 3 or 6 hex digit.

    >>> web2hex('#aaa', force_long=True)
    '#aaaaaa'

    >>> web2hex('#aaaaaa')
    '#aaaaaa'

    >>> web2hex('#aaaa')  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    AttributeError: '#aaaa' is not in web format. Need 3 or 6 hex digit.

    >>> web2hex('pinky')  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: 'pinky' is not a recognized color.

    And color names are case insensitive:

    >>> Color('RED')
    <Color red>

    """
    if web.startswith("#"):
        if LONG_HEX_COLOR.match(web) or (not force_long and SHORT_HEX_COLOR.match(web)):
            return web.lower()
        elif SHORT_HEX_COLOR.match(web) and force_long:
            return "#" + "".join([("%s" % (t,)) * 2 for t in web[1:]])
        raise AttributeError("%r is not in web format. Need 3 or 6 hex digit." % web)

    web = web.lower()
    if web not in COLOR_NAME_TO_RGB:
        raise ValueError("%r is not a recognized color." % web)

    # convert dec to hex:

    return rgb2hex([float(int(v)) / 255 for v in COLOR_NAME_TO_RGB[web]], force_long)


# Missing functions conversion

hsl2hex = lambda x: rgb2hex(hsl2rgb(x))  # noqa: E731
hex2hsl = lambda x: rgb2hsl(hex2rgb(x))  # noqa: E731
rgb2web = lambda x: hex2web(rgb2hex(x))  # noqa: E731
web2rgb = lambda x: hex2rgb(web2hex(x))  # noqa: E731
web2hsl = lambda x: rgb2hsl(web2rgb(x))  # noqa: E731
hsl2web = lambda x: rgb2web(hsl2rgb(x))  # noqa: E731
rgb_equivalence = lambda c1, c2: c1.hex_l == c2.hex_l  # noqa: E731
hsl_equivalence = lambda c1, c2: c1._hsl == c2._hsl  # noqa: E731


class Color(object):
    """Abstraction of a color object.

    Color object keeps information of a color. It can input/output to different
    format (HSL, RGB, HEX, WEB) and their partial representation.

        >>> from ase_notebook.color import Color, HSL

        >>> b = Color()
        >>> b.hsl = HSL.BLUE

    Access values
    -------------

        >>> b.hue  # doctest: +ELLIPSIS
        0.66...
        >>> b.saturation
        1.0
        >>> b.luminance
        0.5

        >>> b.red
        0.0
        >>> b.blue
        1.0
        >>> b.green
        0.0

        >>> b.rgb
        (0.0, 0.0, 1.0)
        >>> b.hsl  # doctest: +ELLIPSIS
        (0.66..., 1.0, 0.5)
        >>> b.hex
        '#00f'

    Change values
    -------------

    Let's change Hue toward red tint:

        >>> b.hue = 0.0
        >>> b.hex
        '#f00'

        >>> b.hue = 2.0/3
        >>> b.hex
        '#00f'

    In the other way round:

        >>> b.hex = '#f00'
        >>> b.hsl
        (0.0, 1.0, 0.5)

    Long hex can be accessed directly:

        >>> b.hex_l = '#123456'
        >>> b.hex_l
        '#123456'
        >>> b.hex
        '#123456'

        >>> b.hex_l = '#ff0000'
        >>> b.hex_l
        '#ff0000'
        >>> b.hex
        '#f00'

    Convenience
    -----------

        >>> c = Color('blue')
        >>> c
        <Color blue>
        >>> c.hue = 0
        >>> c
        <Color red>

        >>> c.saturation = 0.0
        >>> c.hsl  # doctest: +ELLIPSIS
        (..., 0.0, 0.5)
        >>> c.rgb
        (0.5, 0.5, 0.5)
        >>> c.hex
        '#7f7f7f'
        >>> c
        <Color #7f7f7f>

        >>> c.luminance = 0.0
        >>> c
        <Color black>

        >>> c.hex
        '#000'

        >>> c.green = 1.0
        >>> c.blue = 1.0
        >>> c.hex
        '#0ff'
        >>> c
        <Color cyan>

        >>> c = Color('blue', luminance=0.75)
        >>> c
        <Color #7f7fff>

        >>> c = Color('red', red=0.5)
        >>> c
        <Color #7f0000>

        >>> print(c)
        #7f0000


    Recursive init
    --------------

    To support blind conversion of web strings (or already converted object),
    the Color object supports instantiation with another Color object.

        >>> Color(Color(Color('red')))
        <Color red>

    Equality support
    ----------------

    Default equality is RGB hex comparison:

        >>> Color('red') == Color('blue')
        False
        >>> Color('red') == Color('red')
        True
        >>> Color('red') != Color('blue')
        True
        >>> Color('red') != Color('red')
        False

    But this can be changed:

        >>> saturation_equality = lambda c1, c2: c1.luminance == c2.luminance
        >>> Color('red', equality=saturation_equality) == Color('blue')
        True

    """

    _hsl = None  # internal representation

    def __init__(self, color=None, equality=None, **kwargs):
        """Initialise class."""
        if isinstance(color, Color):
            self.web = color.web
        else:
            self.web = color if color else "black"

        self.equality = equality or rgb_equivalence

        for k, v in kwargs.items():
            setattr(self, k, v)

    #
    # Get
    #

    @property
    def hsl(self):
        """Get value."""
        return tuple(self._hsl)

    @property  # noqa: A003
    def hex(self):
        """Get value."""
        return rgb2hex(self.rgb)

    @property
    def hex_l(self):
        """Get value."""
        return rgb2hex(self.rgb, force_long=True)

    @property
    def rgb(self):
        """Get value."""
        return hsl2rgb(self.hsl)

    @property
    def hue(self):
        """Get value."""
        return self.hsl[0]

    @property
    def saturation(self):
        """Get value."""
        return self.hsl[1]

    @property
    def luminance(self):
        """Get value."""
        return self.hsl[2]

    @property
    def red(self):
        """Get value."""
        return self.rgb[0]

    @property
    def green(self):
        """Get value."""
        return self.rgb[1]

    @property
    def blue(self):
        """Get value."""
        return self.rgb[2]

    @property
    def web(self):
        """Get value."""
        return hex2web(self.hex)

    #
    # Set
    #

    @hsl.setter
    def hsl(self, value):
        """Set value."""
        self._hsl = list(value)

    @rgb.setter
    def rgb(self, value):
        """Set value."""
        self.hsl = rgb2hsl(value)

    @hue.setter
    def hue(self, value):
        """Set value."""
        self._hsl[0] = value

    @saturation.setter
    def saturation(self, value):
        """Set value."""
        assert 0 <= value <= 1, "Saturation must be between 0 and 1."
        self._hsl[1] = value

    @luminance.setter
    def luminance(self, value):
        """Set value."""
        assert 0 <= value <= 1, "Lightness must be between 0 and 1."
        self._hsl[2] = value

    @red.setter
    def red(self, value):
        """Set value."""
        _, g, b = self.rgb
        self.rgb = (value, g, b)

    @green.setter
    def green(self, value):
        """Set value."""
        r, _, b = self.rgb
        self.rgb = (r, value, b)

    @blue.setter
    def blue(self, value):
        """Set value."""
        r, g, _ = self.rgb
        self.rgb = (r, g, value)

    @hex.setter  # noqa: A003
    def hex(self, value):
        """Set value."""
        self.rgb = hex2rgb(value)

    @hex_l.setter
    def hex_l(self, value):
        """Set value."""
        self.rgb = hex2rgb(value)

    @web.setter
    def web(self, value):
        """Set value."""
        self.hex = web2hex(value)

    #
    # Convenience
    #

    def __str__(self):
        """Get string."""
        return "%s" % self.web

    def __repr__(self):
        """Get representation."""
        return "<Color %s>" % self.web

    def __eq__(self, other):
        """Test equality."""
        if isinstance(other, Color):
            return self.equality(self, other)
        return NotImplemented


def lighten_webcolor(web_color, fraction):
    """Move a web color towards white by a fraction, or black if the fraction is negative."""
    if fraction == 0:
        return web_color

    import numpy as np

    rgb = np.array(web2rgb(web_color))
    if fraction > 0:
        white = np.array([1.0, 1.0, 1.0])
        rgb = rgb + (white - rgb) * fraction
    else:
        black = np.array([0.0, 0.0, 0.0])
        rgb = rgb + (black - rgb) * abs(fraction)
    return rgb2web(rgb)
