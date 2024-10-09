# with a structure like
# - main.ipynb      <- add lines here
# - src/            <- this git repo
#   - CSVimporter
#   - PlotLib

import sys
sys.path.append('src/')
from PlotLib.Plotting import *
from CSVimporter.importer import load_run