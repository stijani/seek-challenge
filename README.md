# Recruitment Challeng - Data Scientist
### This repository contains my solution to the take home challenge from seek.

**Files and directories:**

`./notebooks`: Contains notebooks used for running code and explaining though process
- `data-cleaning-and-preprocessing.ipynb` -> used for data cleaning and processing. The resulting dataframes where store in `./data/processed`
- `modelling-data-excluding-content-field.ipynb` -> Used to train an MLP and logistic regression model on the Job advert data set. The modelling in this notebook exludes the `content` variable (a child element of the `metedata` column) in the original data.
- `modelling-data-excluding-content-field.ipynb` -> does the same as the prior file but its analysis and modelling included the `content` variable.
- `event-data-exploration.ipynb` -> explores and visualises the event data set

`./data`: Contains both the raw and process data

`./src`: Contains all source files

- `models.py` -> mlp model implementation 
- `vectorizer.py` -> tfidf vectorization implementation
- `helpers` -> helper functions for data cleaning, visualization e.t.c
- `constants` -> contains all macros variable

`./saved_models` -> Contains models saves as .h5 files

`./powerpoint-slides` -> Conaints the final power power point slides


