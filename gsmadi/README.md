# H&T Kinetic Component Analysis (KCA) Assignment

## Getting started

```
virtualenv --python python3.7 ./venv
source venv/bin/activate
pip install -r requirements.txt
```

## Execute unit tests

`pytest`

## Execute linter

`flake8 <FILENAME>`

## Methodology

- We first explore how KCA fits to gold futures contract price to see how good the fit is for in-sample data
- We then take a look at the relationship between position and velocity produced by the KCA fit
- We then apply a trading class based on KCA to make price forecasts and trade decisions against gold futures price to see how well it performs
- Finally, we make some conclusions and how to better obtain results next time

## KCA Trading Algorithm Design

We fit our KCA trading algorithm with 360 days worth of data. We select a year worth of data given thats the resolution we have plus it captures four quarters worth of price movements.

We continously feed in new observations from our test sample to update the KCA fit to take into consideration recent price behavior. We essentially roll the 360 day window on our fit given new observations from our test sample set.