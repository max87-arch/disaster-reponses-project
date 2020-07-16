import sys

import joblib
import nltk
import pandas as pd
import sqlalchemy as sql
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline as PipelineImb
from lightgbm.sklearn import LGBMClassifier
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


def load_data(database_filepath):
    """
    The function loads data from SQLLite db and  it returns X, y and column names of targets (y)
    :param database_filepath: path to SQLLite db file
    :return: X, y and column names of targets
    """
    engine = sql.create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_query("SELECT * from message_with_categories", con=engine)

    X = df['message']
    y = df[['earthquake', 'infrastructure_related',
            # 'child_alone', We remove this category because there aren't rows with this cateogory
            'other_infrastructure', 'other_aid', 'medical_products', 'fire',
            'refugees', 'aid_related', 'shops', 'tools', 'medical_help',
            'direct_report', 'money', 'aid_centers', 'buildings', 'missing_people',
            'hospitals', 'water', 'search_and_rescue', 'military',
            'electricity', 'cold', 'request', 'offer', 'food', 'floods', 'clothing',
            'storm', 'transport', 'weather_related', 'shelter', 'security', 'related',
            'other_weather', 'death']]

    return X, y, y.columns.tolist()


def tokenize(text):
    """
    This function tokenizes the input text, it removes all stopwords and it lemmatizes the words.
    :param text: a string of text
    :return: a list of words
    """
    word_list = nltk.word_tokenize(text.lower())
    only_words_list = [word for word in word_list if word.isalnum()]
    new_words = [nltk.WordNetLemmatizer().lemmatize(word) for word in only_words_list if
                 word not in stopwords.words('english')]

    return new_words


def build_model(search_best_params=True):
    """
    This function initializes the model used to classify the targets.
    The dataset is imbalanced. Thus function combines two strategies to handle the dataset.
    One classifier uses the ADASYN algorithm to oversampling the minority class.
    The other uses the class weight to compensate for the imbalanced dataset.
    To determinate the correct output, the function uses a Voting Classifier.
    The data is preprocessing in two ways.
    The first is to create the Tf-Idf vector, reduced using the LSA algorithm.
    The second is to identify the most relevant topics using the LDA algorithm.
    At the end of the two different pipelines, the features are united.
    To optimize all configurable parameters the function uses Grid Search.

    :param search_best_params: a boolean value that skip Grid Search
    :return: the initialized model
    """
    classifier_handles_smote_data = PipelineImb([
        ('resample_data', ADASYN(random_state=23, n_jobs=-1)),
        ('rfc', LGBMClassifier(random_state=34, n_jobs=-1))
    ])

    classifier_handles_weight_class = Pipeline([
        ('rfc', LGBMClassifier(random_state=34, class_weight={1: 100, 0: 1}, n_jobs=-1))
    ])

    voting_classifer = VotingClassifier(estimators=[
        ('smote_estim', classifier_handles_smote_data),
        ('weight_estim', classifier_handles_weight_class)
    ], weights=[10, 1], voting='hard')

    pipe = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('combined_feature', FeatureUnion([
            ('std_analysis', Pipeline([
                ('tfidf', TfidfTransformer()),
                ('lsa', TruncatedSVD(random_state=42))
            ])),
            ('lda_analysis', Pipeline([
                ('lda', LatentDirichletAllocation(n_jobs=-1, random_state=42)),
            ])),
        ])),
        ('multioutput', MultiOutputClassifier(voting_classifer))
    ])

    if search_best_params:
        parameters = {
            'vect__max_df': (0.5, 1.0),
            'combined_feature__std_analysis__lsa__n_components': (100, 150),
            'combined_feature__lda_analysis__lda__n_components': (1, 2, 3)
        }

        return GridSearchCV(
            estimator=pipe,
            param_grid=parameters,
            scoring='recall_macro',
            verbose=5
        )

    return pipe


def evaluate_model(model, X_test, Y_test, category_names):
    """
    The function prints precision score, recall score, f1-score for each target.
    :param model: the trained model
    :param X_test: the data to classify
    :param Y_test: the true results related to X_test
    :param category_names: the category names of target
    :return: none
    """
    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, columns=Y_test.columns)
    for column in category_names:
        print(column)
        print(classification_report(Y_test[column], y_pred_df[column]))


def save_model(model, model_filepath):
    """
    The function saves the model
    :param model: the trained model
    :param model_filepath: the output path
    :return: none
    """
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) >= 3:
        database_filepath, model_filepath = sys.argv[1], sys.argv[2]
        if len(sys.argv) > 3:
            skip = True
        else:
            skip = False
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model(not skip)

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
