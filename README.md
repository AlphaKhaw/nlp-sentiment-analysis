NLP Sentiment Analysis
# nlp-sentiment-analysis

---

### Table of Contents
You're sections headers will be used to reference location of destination.

- [Description](#description)
- [References](#references)

---

## Description

This respository aims to extract sentiments from Google business reviews scraped from a previous repository: <https://github.com/AlphaKhaw/Google_Business_Reviews>

Firstly, the data undergoes text preprocessing using natural language processing techniques. Through feature engineering, the features are extracted from preprocessed text to fit into list of models.

Binary classifiers are categorised into a dictionary of methods:
- Naive Bayes Classifier
- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree Classifier
- Random Forest Classifier
- K-Nearest Neighbors

Perfomance of each classifier is evaluated using the Accuracy, Precision and Recall metrics.

![Binary Classifier Performance](https://user-images.githubusercontent.com/87654386/154449915-237f484a-9c38-4f67-a15f-38ed2ffd6ef4.png)

## Technologies

- Spyder (Python 3.7)
- Pandas
- Scikit-Learn
- NLTK
- Regular Expression (RegEx)
- tqdm

[Back To The Top](#google-business-reviews)

---

## References

NLTK documentation: <https://www.nltk.org/api/nltk.html>
Scikit-Learn documentation: <https://scikit-learn.org/stable/modules/classes.html>


[Back To The Top](#nlp-sentiment-analysis)
