import re
import pandas as pd
from tqdm import tqdm
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

# ML methods
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
# Logistic Regression
from sklearn.linear_model import LogisticRegression
# Support Vector Machines
from sklearn.svm import LinearSVC
# Decision Trees
from sklearn.tree import DecisionTreeClassifier
# Random Forest
from sklearn.ensemble import RandomForestClassifier
# K-Nearest Neighbours
from sklearn.neighbors import KNeighborsClassifier

class DataPreparationMethods:
    def __init__(self, data, content_series, rating_series):
        self.data = data
        self.content_series = content_series
        self.rating_series = rating_series
        self.score = []
        self.processed = []
        
        self.get_review_dataframe()
        self.apply_methods()
        self.get_processed_df()
        
    def get_review_dataframe(self):
        self.review = self.data[[self.content_series, self.rating_series]]
        return self.review
    
    def score_classification(self):
        for index in range(len(self.review)):
            if self.review.loc[index, self.rating_series] < 3:
                self.score.append(0)
            else:
                self.score.append(1)
        self.review['score'] = self.score
        return self.review
            
    def remove_duplicates(self):
        return self.review.drop_duplicates(subset={self.content_series}, keep='first', inplace=False)
    
    def drop_nan(self):
        return self.review.dropna()
    
    def remove_contractions(self, review):
        phrase = re.sub(r"won't", "will not", str(review))
        phrase = re.sub(r"can\'t", "can not", str(review))
        phrase = re.sub(r"n\'t", " not", str(review))
        phrase = re.sub(r"\'re", " are", str(review))
        phrase = re.sub(r"\'s", " is", str(review))
        phrase = re.sub(r"\'d", " would", str(review))
        phrase = re.sub(r"\'ll", " will", str(review))
        phrase = re.sub(r"\'t", " not", str(review))
        phrase = re.sub(r"\'ve", " have", str(review))
        phrase = re.sub(r"\'m", " am", str(review))
        return phrase
    
    def remove_special_characters(self, review):
        return re.sub('[^a-zA-Z]', ' ', review)
    
    def lowercase(self, review):
        return review.lower()
    
    def tokenization(self, review):
        return review.split()
    
    def lemmatization(self, review):
        lemmatizer = WordNetLemmatizer()
        review = [lemmatizer.lemmatize(word, 'v') for word in review if not word in set(stopwords.words('english'))]
        review = [lemmatizer.lemmatize(word, 'n') for word in review if not word in set(stopwords.words('english'))]
        review = [lemmatizer.lemmatize(word, 'a') for word in review if not word in set(stopwords.words('english'))]
        review = " ".join(review)
        return review
    
    def preprocessing(self, review):
        review = self.remove_contractions(review)
        review = self.remove_special_characters(review)
        review = self.lowercase(review)
        review = self.tokenization(review)
        review = self.lemmatization(review)
        #review = " ".join(review)
        return review
    
    def apply_methods(self):
        self.review = self.score_classification()
        self.review = self.remove_duplicates()
        self.review = self.drop_nan()
        for index, row in tqdm(self.review.iterrows()):
            processed_text = self.preprocessing(row[self.content_series])
            self.processed.append(processed_text)
        return self.processed
    
    def get_processed_df(self):
        self.processed_df = pd.DataFrame({'x' : self.processed, 
                                          'y': self.review['score']})
        
class BinaryClassifier:
    def __init__(self, df, method):
        self.df = df
        self.method = method
        self.cv = CountVectorizer(ngram_range=(1,3), max_features = 5000)
        self.X = df['x']
        self.y = df['y']
        self.accuracy, self.precision, self.recall = {}, {}, {}
        self.models = {
                    'Naive Bayes': GaussianNB(),
                    'Logistic Regression': LogisticRegression(),
                    'Support Vector Machines': LinearSVC(),
                    'Decision Trees': DecisionTreeClassifier(),
                    'Random Forest': RandomForestClassifier(),
                    'K-Nearest Neighbor': KNeighborsClassifier()
                    }
        
        self.vectorize_X_variable()
        self.train_test_split()
        self.run_method()
        self.get_result_df()
        
    def vectorize_X_variable(self):
        self.X = self.cv.fit_transform(self.X).toarray()
        return self.X 
    
    def train_test_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.25, random_state = 0)
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def run_method(self):
        if self.method == 'All':
            for key in self.models.keys():
                self.models[key].fit(self.X_train, self.y_train)
                predictions = self.models[key].predict(self.X_test)
                self.accuracy[key] = accuracy_score(predictions, self.y_test)
                self.precision[key] = precision_score(predictions, self.y_test)
                self.recall[key] = recall_score(predictions, self.y_test)
        else:
            self.models[self.method].fit(self.X_train, self.y_train)
            predictions = self.models[self.method].predict(self.X_test)
            self.accuracy[self.method] = accuracy_score(predictions, self.y_test)
            self.precision[self.method] = precision_score(predictions, self.y_test)
            self.recall[self.method] = recall_score(predictions, self.y_test)

    def get_result_df(self):
        if self.method == 'All':
            self.result_df = pd.DataFrame(index=self.models.keys(), 
                                          columns=['Accuracy', 'Precision', 'Recall'])
            self.result_df['Accuracy'] = self.accuracy.values()
            self.result_df['Precision'] = self.precision.values()
            self.result_df['Recall'] = self.recall.values()
        
        else:
            self.result_df = pd.DataFrame(index=[self.method], 
                                          columns=['Accuracy', 'Precision', 'Recall'])
            self.result_df['Accuracy'] = self.accuracy.values()
            self.result_df['Precision'] = self.precision.values()
            self.result_df['Recall'] = self.recall.values()