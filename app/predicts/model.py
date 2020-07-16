import joblib


def get_model():
    """
    The function returns the model
    :return: Model
    """
    # load model
    model = joblib.load("../models/classifier.joblib")
    return model
