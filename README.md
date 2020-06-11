# Ticket to Ride Where?

Ticket to Ride Where? is a machine learning model that predicts public transit ridership levels in Illinois Census tracts. The model identifies Census tracts that would use public transportation the most given an increase in transit access. This model can be used as a tool to inform decisions about where to prioritize investments in transit infrastructure.

## Installation

In your preferred directory, clone the repository using git:

```bash
git clone https://github.com/ndignazio/transit-ml.git
```
Install required packages in a virtual environment:

```bash
pip3 install -r requirements.txt
```
## Usage
Run using an archived version of the data and best model:
```bash
python3 main.py -m
```
Run using an archived version of the data:
```bash
python3 main.py -d
```
Run using no archives:
```bash
python3 main.py
```


## Authors
The authors of this repository are Nathan Dignazio, Mike Feldman, and Nguyen Luong, three graduate students at the University of Chicago.

## License
[MIT](https://choosealicense.com/licenses/mit/)

