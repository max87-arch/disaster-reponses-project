import joblib


def get_model(path_model="../models/classifier.joblib"):
    """
    The function returns the model
    :param path_model: path to model
    :return: Model
    """
    # load model
    model = joblib.load(path_model)
    return model
