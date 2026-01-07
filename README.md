# VTAnalysis
Python scripts to analyze cosmic telescope data

-------------------------------------------------
## Dependancies

numpy, scipy, pandas, matplotlib

## Usage
The library file has a `main` function which shows one example usage. It can be run from the command line:
```
python analysis_lib.py
```
This creates 3 plots: the signals from WaveDump, raw histograms, final plot w/ residuals

The GUI can be started from the command line with:
```
python analysisGui_v2.py
```
Fill in the requested fields, then `analysis_lib.main` will run with the specified values.
