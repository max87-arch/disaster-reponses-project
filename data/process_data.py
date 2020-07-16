import sys

import pandas as pd
import sqlalchemy as sql


# Extract data
def load_data(messages_filepath, categories_filepath):
    """
    the function creates two DateFrame from two input arguments, and it merges them in one DateFrame
    :param messages_filepath: the path of messages CSV file
    :param categories_filepath: the path of categories CSV file
    :return: one DataFrame with messages and categories merged
    """
    categories_df = pd.read_csv(categories_filepath)
    disasters_df = pd.read_csv(messages_filepath)
    df = pd.merge(categories_df, disasters_df, on='id')
    return df


# Transform data
def clean_data(df):
    """
    The function cleans the DataFrame. In particular, converts the encoded categories. Each category becomes a
    boolean column.
    :param df: the DataFrame to clean
    :return: the cleaned DataFrame
    """
    values_columns = list()
    categories = {'id'}
    for row_categories_encoded in df.itertuples():
        row = {'id': row_categories_encoded.id}
        for category_encoded in row_categories_encoded.categories.split(';'):
            category, value = category_encoded.split('-')
            row[category] = (value == '1')
            categories.add(category)
        values_columns.append(row)
    categories_df = pd.DataFrame(values_columns, columns=categories, index=df.index)
    df = df.merge(categories_df, on='id')
    df.drop(columns=['original', 'categories'], inplace=True)
    return df


# Load data
def save_data(df, database_filename):
    """
    The function stores the data inside a SQLite Database
    :param df: The DataFrame to store
    :param database_filename: the path of database
    :return: None
    """
    engine = sql.create_engine("sqlite:///" + database_filename)
    df.to_sql("message_with_categories", engine, if_exists="replace", index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
