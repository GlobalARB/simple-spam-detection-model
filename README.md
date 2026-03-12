# Simple Spam Detection Model

An educational implementation of two fundamental machine learning classifiers for email spam detection: k-Nearest Neighbors (k-NN) and Gaussian Naive Bayes. Built from scratch in Python to demonstrate core ML concepts without relying on high-level library abstractions.

This project is intended as a learning exercise in classical machine learning. The implementations prioritize clarity and understanding over production-level performance.

## Overview

The notebook processes the Enron-1 email dataset and builds two classifiers from first principles:

**Naive Bayes Classifier**
- Gaussian Naive Bayes implementation with configurable train/test split
- Parameter estimation (mean, variance) per class
- Log-probability prediction to avoid numerical underflow
- Accuracy evaluation across multiple split ratios

**k-Nearest Neighbors (k-NN) Classifier**
- Supports L1 (Manhattan), L2 (Euclidean), and L-infinity (Chebyshev) distance metrics
- Configurable k values and train/test splits
- Comparison against scikit-learn's KNeighborsClassifier for validation

## Dataset

Uses the [Enron-1 spam dataset](http://nlp.cs.aueb.gr/software_and_datasets/Enron-Spam/index.html), containing labeled ham and spam emails as raw text files. Emails are preprocessed with tokenization, stemming (Porter Stemmer), and bag-of-words feature extraction via `DictVectorizer`.

## Requirements

- Python 3.8+
- See `requirements.txt` for dependencies

```
pip install -r requirements.txt
```

## Usage

Open and run `SpamClassifier.ipynb` in Jupyter:

```
jupyter notebook SpamClassifier.ipynb
```

The notebook walks through data loading, preprocessing, model training, evaluation, and visualization of results across different hyperparameter configurations.

## Results

Both classifiers are evaluated across split ratios (0.3, 0.5, 0.7, 0.9) with accuracy plotted for training and test sets. The k-NN classifier is additionally benchmarked across k values (1, 2, 5, 10) and compared against scikit-learn's implementation.

## License

MIT License. See [LICENSE](LICENSE) for details.
