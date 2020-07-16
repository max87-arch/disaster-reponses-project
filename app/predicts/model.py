import joblib


def get_model():
    # load model
    model = joblib.load("../models/classifier.joblib")
    return model
