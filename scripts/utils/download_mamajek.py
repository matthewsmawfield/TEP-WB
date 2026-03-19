import urllib.request
import pandas as pd
import numpy as np

url = "https://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt"
print("Downloading Mamajek table...")
urllib.request.urlretrieve(url, "data/raw/EEM_dwarf_UBVIJHK_colors_Teff.txt")
print("Done.")
