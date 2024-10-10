# with a structure like
#   - main.ipynb            <- add lines here
#   - PythonHelpers/        <- this git repo
#       - CSVimporter/
#           - importer.py
#           - runSettings.py
#       - PlotLib/
#           - Plotting.py

import sys
sys.path.append('./PythonHelpers')
from PlotLib.Plotting import *
from CSVimporter.importer import load_run