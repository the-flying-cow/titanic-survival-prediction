# Titanic Survival Prediction ML Project
A simple machine learning project built using Logistic Regression model in Jupyter on the Titanic Survival prediction dataset.

## Description
The sinking of the Titanic is one of the most infamous shipwrecks in history.
On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.
While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.
This project trains and evaluates a Logistic Regression model on the given passenger dataset for the Titanic Survival Prediction.
In this challenge, we build a predictive model that answers the question: “What sorts of people were more likely to survive?” using passenger data given to us.

## Directory Structure
```bash
src/
│── data_load.py        # functions to load Titanic dataset
│── eda.py              # eda visualizations and summary stats
│── pre_process.py      # preprocessing steps (missing values, encoding, scaling)
│── variables.py        # feature and target definitions
│── model.py            # model training, prediction, and reporting
│── main.py             # orchestrates the pipeline
titanic_dataset/        # dataset folder
README.md               # documentation
requirements.txt        # specifies the dependencies
```
## Setup
```bash
git clone https://github.com/the-flying-cow/titanic-survival-prediction.git
cd titanic-survival-prediction
```

Create a virtual environment:
```bash
python -m venv .venv
venv\Scripts\activate
```
## Run the project

In your terminal/command prompt, simply navigate to the root/ project folder and execute the following 
```bash
pip install -r requirements.txt
```
Inside the src folder, run the following
```bash
python main.py
```
