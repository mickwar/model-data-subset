## Modeling subsets of data

This repository attempts to address the question, "Does including bad data from one category of data affect the fidelity of another category of data?"
By *bad*, we might mean data which is skewed, abnormal, or hard to predict.
And by *category*, we mean a factor level for some covariate.

How would a model handle situations where the relationship between inputs and outputs varies widely depending on the factor level?
What if the input-output relationship were the same across factor levels, but the marginal variance of covariates from one factor level were 10 times that of another?
Or rather, what if the covariates were sampled from different distributions per factor level?
In these cases, it may be helpful to know whether the model breaks with the inclusion of the peculiar data subset&mdash;the subset associated with the offending factor level, the so-called *bad* data.

We explored several models under a variety of data configurations.

The models are:
- Linear regression
- XGBoost
- LightGBM
- Feedforward neural network

The data contain normally distributed covariates, some with low variance, some with high, some with a simple input-output relationship, some with no relationship.

We will look at a model which includes only the supposedly *good* data and compare the quality against a model which also includes some *bad* data.
We want to examine the predictions on the *good* subset only, and see how the models differ when the *good* data is trained alongside *bad* data.

### Set up a Python3 virtual environment

```bash
virtualenv -p python3 env       # intialize python3 virtual environment
source env/bin/activate         # activate the virtual environment
deactivate                      # deactivate the virtual environment
```

```bash
pip freeze > requirements.txt   # create for the first time
pip freeze >! requirements.txt  # overwrite file
```
