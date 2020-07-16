from data import disasters as df
from predicts import disaster_model as model


def predict_categories(query):
    """
    This function classifies the text in input
    :param query: the text to classify
    :return: the categories of the text
    """
    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    return classification_results
