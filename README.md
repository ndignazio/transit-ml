# Ticket to Ride Where?

Ticket to Ride Where? is a machine learning project that predicts public transit ridership rates in Illinois Census tracts. The project generates a model that identifies Census tracts that would use public transportation the most given an increase in transit access. The hope is that this model can be used as a tool to inform community-based decision-making about where to prioritize investments in public transit infrastructure.

## Installation

In your preferred directory, clone the repository using git:

```bash
git clone https://github.com/ndignazio/transit-ml.git
```
Install required packages in a virtual environment:

```bash
pip3 install -r requirements.txt
```

## Structure

__main.py__ governs the text-based UI of this repository. In the terminal, users may choose to rely on archived versions of the data and/or best model or run the entire program from scratch (which will take several hours).

__pipeline.py__ contains useful helper functions to get ACS data, impute missing values, run grid search over multiple pipelines, and identify and record information about best-performing models.

__download.py__ and __data_wrangling.py__ get data from ACS and the WalkScore API, merge it with files in __data_sources__, and performs necessary cleaning before returning a DataFrame ready for modeling.

__model_selection.py__ takes the DataFrame, splits it into training and testing sets, and runs a grid search over pre-selected regression models and hyperparameters to identify the best model, which is saved to best_model.pkl.

__recommend.py__ produces DataFrames of 1) Census tracts recommended for increased transit investment based on the results of the best model and 2) Census tracts recommended for further inspection based on a large positive difference between the best model's predictions and the tract's actual ridership rates.

__CENSUS_DATA_COLS.json__ contains a dictionary mapping ACS 5-year table ID's to column labels with information about what each table contains.

The __pickle_files__ folder contains .pkl files generated from other files in the repository relating to model selection and Census tracts recommended from recommend.py.

The __data_sources__ folder contains shapefiles and LEHD Origin-Destination Employment Statistics (LODES).

## Usage
To run using an archived version of the data and best model:
```bash
python3 main.py -m
```
To run using an archived version of the data:
```bash
python3 main.py -d
```
To run using no archives:
```bash
python3 main.py
```


## Authors
The authors of this repository are Nathan Dignazio, Mike Feldman, and Nguyen Luong, three graduate students at the University of Chicago.

## License
[MIT](https://choosealicense.com/licenses/mit/)