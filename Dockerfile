FROM ubuntu:20.04

RUN apt-get update -y && \
    apt-get install -y python3-pip python3-dev

WORKDIR /app

RUN pip3 install scikit-learn==0.22 imbalanced-learn==0.6 scikit-optimize pandas plotly SQLAlchemy Flask nltk joblib lightgbm

COPY . /app

RUN python3 data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

RUN python3 models/train_classifier.py data/DisasterResponse.db models/classifier.joblib skip

EXPOSE 3001

WORKDIR /app/app

ENTRYPOINT [ "python3" ]
CMD ["run.py"]
