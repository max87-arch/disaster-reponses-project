import json

import plotly
from flask import Flask
from flask import render_template, request, jsonify
from graphs.plot import plot_genre_bar, plot_categories_bar, plot_categories_polar
from predicts.categories import predict_categories

app = Flask(__name__)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # create visuals
    graphs = [
        plot_genre_bar(),
        plot_categories_bar(),
        plot_categories_polar()
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    classification_results = predict_categories(query)

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
