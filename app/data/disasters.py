import pandas as pd
from sqlalchemy import create_engine


def get_data():
    """
    The function returns the data
    :return: DataFrame data
    """
    # load data
    engine = create_engine('sqlite:///../data/DisasterResponse.db')
    df = pd.read_sql_table('message_with_categories', engine)
    return df
