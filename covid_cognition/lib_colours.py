# -----------------------------------------------------------------------------
# This lib file contains helper functions for working with colours in plotly
# and matplotlib. Again, this should be cleaned up but it's low on the 
# priority list, for now.
#
# Include custom colourmaps sequences generated from
#   https://colorbrewer2.org/
#
# -----------------------------------------------------------------------------
# cwild 2021-04-15

from _plotly_utils.colors import unlabel_rgb
import numpy as np

_DIVERGING_C1 = [
    (64,0,75),
    (118,42,131),
    (153,112,171),
    (194,165,207),
    (231,212,232),
    (217,240,211),
    (166,219,160),
    (90,174,97),
    (27,120,55),
    (0,68,27),
]

_DIVERGING_C2 = [
    (103,0,31),
    (178,24,43),
    (214,96,77),
    (244,165,130),
    (253,219,199),
    (209,229,240),
    (146,197,222),
    (67,147,195),
    (33,102,172),
    (5,48,97),
]

_DIVERGING_C3 = [
    (165,0,38),
    (215,48,39),
    (244,109,67),
    (253,174,97),
    (254,224,144),
    (224,243,248),
    (171,217,233),
    (116,173,209),
    (69,117,180),
    (49,54,149)
]

_DIVERGING_C4 = [
    (142,1,82),
    (197,27,125),
    (222,119,174),
    (241,182,218),
    (253,224,239),
    (230,245,208),
    (184,225,134),
    (127,188,65),
    (77,146,33),
    (39,100,25),
]


# _RB_CMAP = create_mpl_cmap(plotly_cmap('diverging', 'RdBu'))
# _PICNIC_CMAP = create_mpl_cmap(plotly_cmap('diverging', 'Picnic'))

def is_rgba(c):
    """ Is the passed color string (in a tuple format) RGBA?
    """
    if isinstance(c, str):
        return c[0:4].lower() == 'rgba'
    else:
        return len(c) == 4

def is_rgb(c):
    """ Is the passed colors string (in a tuyple form) RGB?
    """
    if isinstance(c, str):
        return c[0:4].lower() == 'rgb('
    else:
        return len(c) == 3


def rgb_to_rgba(cmap, alpha=1.0, norm=True):
    """ Convert a string-type colourmap (i.e., a list of color string) 
        from RGB to RGBA.
    """
    if is_rgba(cmap[0]):
        return cmap

    if isinstance(cmap[0], str):
        rgb = np.array([[float(val) for 
            val in c[4:-1].split(',')] for c in cmap], dtype='float32')
    else:
        rgb = np.array(cmap, dtype='float32')

    # If any of the values are > 1, then we assume it's RGB 255 so we have
    # to normalize to 0.0 -> 1.0
    if (rgb > 1.0).any():
        rgb /= 255
    
    rgb = np.c_[rgb, alpha*np.ones([rgb.shape[0],1])]

    return [tuple(rgb[i,:]) for i in range(rgb.shape[0])]

def unlabel_rgbs(cmap):
    from plotly import colors
    return [colors.unlabel_rgb(c) for c in cmap]

def to_RGB_255(cmap, as_str=True):
    from plotly import colors
    cmap = [colors.convert_to_RGB_255(c) for c in cmap]
    if as_str:
        return [colors.label_rgb(c) for c in cmap]
    else:
        return cmap

def plotly_cmap(type_, name_):
    from plotly import colors
    ctype = getattr(colors, type_)
    pmap  = getattr(ctype, name_)
    return pmap

def create_mpl_cmap(cmap, n_steps=10, alpha=1.0):
    """
    """
    from matplotlib.colors import ListedColormap
    from plotly.colors import find_intermediate_color as fic

    ncols = len(cmap)

    if is_rgb(cmap[0]) and alpha is not None:
        cmap = rgb_to_rgba(cmap, alpha=alpha)
    elif isinstance(cmap[0], str):
        cmap = unlabel_rgbs(cmap)

    step = np.arange(0, 1.0, 1.0/n_steps)
    cmap = [fic(cmap[ci], cmap[ci+1], s) for ci in range(ncols-1) for s in step]
    cmap = [c + (alpha,) for c in cmap]

    return ListedColormap(cmap)

D1_CMAP = create_mpl_cmap(_DIVERGING_C1)
D2_CMAP = create_mpl_cmap(_DIVERGING_C2)
D3_CMAP = create_mpl_cmap(_DIVERGING_C3)
D4_CMAP = create_mpl_cmap(_DIVERGING_C4)

