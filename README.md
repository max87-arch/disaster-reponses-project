# Disaster Response Pipeline Project

This project has the objective to create a predictive model capable of classifying text messages and associate one or more categories.
In particular, the dataset contains messages related to disaster scenarios, and they have one or more categories associated.
The project has three steps:

* ETL pipeline: it's to prepare the original dataset to model
* ML pipeline:  it's to build the predictive model
* FLASK Web App: it's to make available the result to everyone.

## Getting start

Before proceeding with installation, the software requires the following dependencies:
```bash
pip3 install scikit-learn==0.22 imbalanced-learn==0.6 scikit-optimize pandas plotly SQLAlchemy Flask nltk joblib
```

The original dataset can be download from (Appen Site)[https://appen.com/datasets/combined-disaster-response-data/].
Please consider that this code runs on the old version of the dataset. This version doesn't include it to avoid license problem.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.joblib`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Some considerations on project
We can observe that the dataset is imbalanced. To manage the dataset, we combine two strategies:
* We use the ADASYN algorithm to oversampling the minority class.
* We use the class weight to compensate for the imbalanced dataset.
To determinate the best approach for the outcome, we use a Voting Classifier. This solution permits us to choose which strategy is best.

An important aspect is how preprocessing the message texts. We unite two features created from message texts:
* The first feature creates the Tf-Idf vector, reduced using the LSA algorithm.
* The second feature identifies the most relevant topics using the LDA algorithm.

To optimize all configurable parameters, we use a Grid Search.
