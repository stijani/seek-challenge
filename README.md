# Recruitment Challenge - Data Scientist
**This repository contains my solution to the take home challenge from Seek. I leveraged the logistic regression and an MLP model to classify jobs into categories (sector) using the job advert data set. I splitted the data into two parts, one that contained the `content` (i.e the full JD) field and one that excluded it. Seperate models were built on each data split (see notebook descriptions below).**

**The idea is that once a job poster enters the abstract text for a job, an ML generated job category is suggested to them. Also, after completing the content section, the category is updated by a more accurate ML model.**

**PS: I pruned the number of classes to 12 rather than 30 in the raw data set. This was done to avoid potential model skewness from class inbalance. The code could easily be extended to include all classes once more data have been added to improve the coverage of the minority categories.**


## Files and directories:

`./notebooks`: Contains notebooks used for running code and explaining thought process.
- `data-cleaning-and-preprocessing.ipynb` -> Used for data cleaning and processing. The resulting dataframes were stored in `./data/processed`
- `modelling-data-excluding-content-field.ipynb` -> Used to train an MLP and logistic regression models on the job advert data set. The modelling in this notebook exludes the `content` field (a child element of the `metedata` column) in the original data.
- `modelling-data-excluding-content-field.ipynb` -> Does the same as the prior file but its analysis and modelling included the `content` variable.
- `event-data-exploration.ipynb` -> explores and visualises the event data set

`./data`: Contains both the raw and processed data. Please get the data files from: https://drive.google.com/drive/folders/1hSuRyqoNWQqmUBCgu6rrkhvWgd4CO9XH?usp=sharing

`./src`: Contains all source files

- `models.py` -> mlp model implementation in tensorflow
- `vectorizer.py` -> tfidf vectorization implementation (SKlearn)
- `helpers` -> helper functions for data cleaning, visualization e.t.c
- `constants` -> contains all macro variables

`./saved_models` -> Contains models saved as .h5 files

`./powerpoint-slides` -> Contains the final power point slides. please get the powerpoint file from: https://docs.google.com/presentation/d/1iK6kIoMaVVgsJsPZSMADwNK_Ul0HgGRr/edit?usp=sharing&ouid=103754789475705413599&rtpof=true&sd=true


