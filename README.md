# Recruitment Challenge - Data Scientist
**This repository contains my solution to the take home challenge from Seek. I leveraged the logistic regression and an MLP model to classify jobs into categories (sector) using the job advert data set. I splitted the data into two parts, one that contained the `content` (i.e the full JD) field and one that excluded it. Seperate models were built on each data split (see notebook descriptions below).**

**The idea is that once a job poster enters the abstract text for a job, an ML generated job category is suggested to them. Also, after completing the content section, the category is updated by a more accurate ML model.**


**Files and directories:**

`./notebooks`: Contains notebooks used for running code and explaining thought process.
- `data-cleaning-and-preprocessing.ipynb` -> used for data cleaning and processing. The resulting dataframes where store in `./data/processed`
- `modelling-data-excluding-content-field.ipynb` -> Used to train an MLP and logistic regression model on the Job advert data set. The modelling in this notebook exludes the `content` variable (a child element of the `metedata` column) in the original data.
- `modelling-data-excluding-content-field.ipynb` -> does the same as the prior file but its analysis and modelling included the `content` variable.
- `event-data-exploration.ipynb` -> explores and visualises the event data set

`./data`: Contains both the raw and process data. Please get the data from https://drive.google.com/drive/folders/1hSuRyqoNWQqmUBCgu6rrkhvWgd4CO9XH?usp=sharing

`./src`: Contains all source files

- `models.py` -> mlp model implementation 
- `vectorizer.py` -> tfidf vectorization implementation
- `helpers` -> helper functions for data cleaning, visualization e.t.c
- `constants` -> contains all macros variable

`./saved_models` -> Contains models saves as .h5 files

`./powerpoint-slides` -> Contains the final power power point slides. please get the powerpoint file from: https://docs.google.com/presentation/d/1iK6kIoMaVVgsJsPZSMADwNK_Ul0HgGRr/edit?usp=sharing&ouid=103754789475705413599&rtpof=true&sd=true


