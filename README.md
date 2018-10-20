# ann-py
CS 280 Programming Assignment 2

OS: macOS Mojave Version 10.14
Python: Version 3.7

Setup:
```
pip install -r requirements.txt
```
or
```
pipenv install
```

Running the ANN:
----
```
python ann_runner.py --data data.csv --labels data_labels.csv --in 354 --h1 300 --h2 250 --out 8 --eta 0.0001 --test-set test_set.csv
```

Running the SVM:
----

```
python svm_runner.py --data data.csv --labels data_labels.csv --test-set test_set.csv
```
