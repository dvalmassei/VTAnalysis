# VTAnalysis
Python scripts to analyze cosmic telescope data

-------------------------------------------------
## Dependancies
-[PyLandau](https://pypi.org/project/pylandau/)
```
pip install pylandau
```

## Usage
The library file has a `main` function which shows one example of how the program opporates. It can be run from the command line:
```
python analysis_lib.py
```
This creates 3 plots: the signals from WaveDump, raw histograms, final plot w/ residuals

The GUI can be started from the command line with:
```
python analysisGui_vs.py
```
Fill in the requested fields, then `analysis_lib.main` will run with the specified values.
