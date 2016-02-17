Machine Learning in Python
==========================

Structure
---------

Courses are avalaible in three formats:

1. Python files in the [python](https://github.com/duchesnay/pylearn-doc/tree/master/python) directory.

2. Ipython notebooks files in the  in the [notebooks](https://github.com/duchesnay/pylearn-doc/tree/master/notebooks) directory.

3. ReStructuredText files in the [rst](https://github.com/duchesnay/pylearn-doc/tree/master/rst) directory.

All notebooks and python files are converted into `rst` format and then assembled together using sphinx.

Build
-----
Build the pdf file:
```
make pdf
```

Build the html files:
```
make html
```
Dependencies
------------

- python 3
- ipython
The easier is to install Anaconda at https://www.continuum.io with python >= 3

- pandoc

For Linux:
```
sudo apt-get install pandoc
```
