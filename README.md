[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

# stochastic-investment-planning

This is a ![](https://img.shields.io/badge/Python-v%203.8-blue) implementation of a two-stage stochastic model used for investment planning under uncertainity. The modified problem is taken from [1]. This particual problem originally appeared in [2]. The problem is first implemented as a determinestic linear program using ![](https://img.shields.io/badge/Pyomo-v%206.4.1-orange) and later solved using ![](https://img.shields.io/badge/mpi--sppy-v%200.10-orange).

## Files and Folders
* :file_folder: 'run_me.ipynb' - The main Jupyter Noteboomk file which solves the problem with explanations
* :file_folder: 'helper_function.py' - Python file containing all necessary functions required to run 'run_me.ipynb'
* :file_folder: 'requirements.txt' - List of required packages for running the code, that can be downloaded directly using ![](https://anaconda.org/anaconda/jupyter/badges/version.svg)


## Installation
The repository can be cloned to the local computer manually or using the following code:
```
git clone https://github.com/nkpanda97/stochastic-investment-planning
```
The required packages can be installed dorectly using 'requirements.txt' file in ![](https://anaconda.org/anaconda/jupyter/badges/version.svg) using the following command:
```
$ conda create --name <env> --file <this file>
$ platform: osx-64
```

#





## References
[1] Klein Haneveld, W. K., van der Vlerk, M. H., & Romeijnders, W. (2020). Stochastic Programming. In Graduate Texts in Operations Research. Springer International Publishing. https://doi.org/10.1007/978-3-030-29219-5 <br>
[2] F.V. Louveaux and Y. Smeers. Optimal investments for electricity generation: a stochastic model and a test problem. In Yu. Ermoliev and R.J-B. Wets, editors, Numerical techniques for stochastic optimization, chapter 24, pages 445–453. Springer-Verlag, Berlin, 1988.
