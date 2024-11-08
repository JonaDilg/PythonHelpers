# with a structure like
#   - main.ipynb            <- add lines here
#   - PythonHelpers/        <- this git repo
#       - CSVimporter/
#           - importer.py
#           - runSettings.py
#       - PlotLib/
#           - Histogramming.py
#           - Plotting.py

import sys
sys.path.append('./PythonHelpers')
import PlotLib.Histogramming as hist
import PlotLib.Plotting as plot
from CSVimporter.importer import load_run