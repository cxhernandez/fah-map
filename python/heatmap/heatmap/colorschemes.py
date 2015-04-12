""" color scheme source data """
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

SCHEMES = ["Blues", "Blues_r", "Reds", "Reds_r", "Greys", "Greys_r", "Greens",
"Greens_r", "Purples", "Purples_r", "Oranges", "Oranges_r", "BuGn", "BuGn_r",
"RdBu", "RdBu_r", "PuBu", "PuBu_r", "OrRd", "OrRd_r"]

schemes = {}
for s in SCHEMES:
    schemes.update({s :  np.rint(255*np.array(sns.color_palette(s,
    256))).astype(int)})
cmap = LinearSegmentedColormap.from_list('mycmap', [(0, "gold"), (0.5, "yellow"),
(1, "white")])
schemes.update({'custom': np.rint(255*cmap(np.linspace(0,1,256))[:,:-1]).astype(int)})

def valid_schemes():
    return schemes.keys()

