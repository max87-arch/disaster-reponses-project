from data import disasters as df
from plotly.graph_objs import Bar, Scatter, Scatterpolar


def plot_genre_bar():
    """
    This function returns a Bar plot that describes Genre data
    :return: Object json for graphs
    """
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    return {
        'data': [
            Bar(
                x=genre_names,
                y=genre_counts
            )
        ],

        'layout': {
            'title': 'Distribution of Message Genres',
            'yaxis': {
                'title': "Count"
            },
            'xaxis': {
                'title': "Genre"
            }
        },

    }


def plot_categories_bar():
    """
    This function returns a Scatter plot that describes categories
    :return: Object json for graphs
    """
    # extract data needed for visuals
    categories_df = df.select_dtypes(['boolean']).drop(
        columns=['related', 'aid_related', 'weather_related', 'direct_report'])

    categories_counts = categories_df.sum()
    categories_names = categories_counts.index.tolist()

    return {
        'data': [
            Scatter(
                y=categories_names,
                x=categories_counts,
                mode='markers'
            )
        ],

        'layout': {
            'title': 'Distribution of Categories',
            'yaxis': {
                'title': "Category"
            },
            'xaxis': {
                'title': "Count"
            },
            'height': '700'
        },

    }


def plot_categories_polar():
    """
    This function returns a Scatter Polar plot that describes categories
    :return: Object json for graphs
    """
    # extract data needed for visuals
    categories_df = df.select_dtypes(['boolean']).drop(
        columns=['related', 'aid_related', 'weather_related', 'direct_report'])

    categories_counts = categories_df.sum()
    categories_names = categories_counts.index.tolist()

    return {
        'data': [
            Scatterpolar(
                theta=categories_names, r=categories_counts,
                mode='markers'
            )
        ],

        'layout': {
            'title': 'Polar of Categories',
            'height': '700'
        },

    }
