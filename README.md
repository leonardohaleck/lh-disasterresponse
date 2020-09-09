# Disaster Response Pipeline Project


### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Licensing, Authors, and Acknowledgements](#licensing)
5. [Instructions](#instructions)

## Installation <a name="installation"></a>

The libraries used in these notebooks are: Sklearn, Numpy, Pandas, Flask, Plotly and GUnicorn.
The code was written using Python 3.

## Project Motivation<a name="motivation"></a>

The main objective of this project was to create a Disaster Response Classifier that classifies
a disaster message according to its content. 
It helps to direct each message to the organization responsible for its category.

## File Descriptions <a name="files"></a>

• train_classifier.py: ML pipeline that trains classifier and saves
• process_data.py: ETL pipeline that cleans data and stores in database
• app.py: creates the charts and initialize the application in Flask

• disaster_messages: contains the disaster messages
• disaster_categories: contains the disaster messages's categories

• master.html: contains the HTML input's page code
• go.html: contains the HTML result's page code

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to Udacity and Figure Eight for the data.


## Instructions <a name="instructions"></a>

0. If DisasterResponse.db and classifier.pkl already exist, you can run run.py directly.

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`
