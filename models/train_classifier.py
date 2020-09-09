import sys
import nltk
nltk.download(['punkt', 'wordnet'])
import re
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import joblib

from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

def load_data(database_filepath):

    '''
    INPUT:
        database_filepath: SQL Database's filepath

    OUTPUT:
        X: disaster messages
        Y: disater categories
        category_names: categories's labels
    '''

    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df.message.values
    Y = df[df.columns[4:]].values
    category_names = list(df.columns[4:])

    return X, Y, category_names

def tokenize(text):

    '''
    INPUT:
        text: messages to tokenize

    OUTPUT:
        clean_tokens: tokenized messages
    '''

    # Cleaning with regex and transforming text to lower case
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Tokenization
    tokens = word_tokenize(text)

    # Lemmatization
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []

    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():

    '''
    OUTPUT:
        cv: result of GridSearchCV
    '''

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(OneVsRestClassifier(LinearSVC(random_state = 0))))
    ])
    parameters = {
                'tfidf__smooth_idf':[True, False],
                'clf__estimator__estimator__C': [1, 2, 4, 8]
             }
    cv = GridSearchCV(pipeline, param_grid=parameters, cv = 5)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    input:
        model: the model you want to evaluate
        X_test: features of testing set
        Y_test: target of testing set
        category_names: a list containing the names of categories

    output:
        The report of precision, recall, F1-Score and Accuracy for all categories    
    '''

    # Evaluating Precision, Recall, F1-Score
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names = category_names))
    print('---------------------------------')

    # Evaluating Accuracy
    for i in range(Y_test.shape[1]):
        print('%25s accuracy : %.2f' %(category_names[i], accuracy_score(Y_test[:,i], Y_pred[:,i])))

def save_model(model, model_filepath):
    '''
    input:
        model: the model you want to save
        model_filepath: file path you want to save your model

    output:
        A model saved in file path
    '''

    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
