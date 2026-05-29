# Chinese Cosmetics Review Sentiment Analysis

This project analyzes Chinese e-commerce cosmetics reviews and builds machine-learning models to classify positive and negative customer feedback.

## Project Overview

The project uses Chinese cosmetics review data from multiple brands and performs a full text-mining and sentiment-classification workflow.

Main steps include:

- merging review datasets from multiple Excel files
- cleaning Chinese review text and removing HTML artifacts
- tokenizing Chinese text with `jieba`
- building stopword lists and beauty-domain phrase rules
- extracting TF-IDF and word-count features
- training and comparing multiple classification models
- analyzing misclassified samples
- using LIME to explain model predictions

## Models Used

The project compares several machine-learning models:

- Logistic Regression
- Random Forest
- Multinomial Naive Bayes
- Voting Classifier
- Stacking Classifier

Evaluation metrics include accuracy, precision, recall, F1 score, classification report, and confusion matrix.

## Repository Files

- `姚隆浚-week 7.ipynb`  
  Main Jupyter Notebook containing the full analysis workflow.

- `requirements.txt`  
  Python dependencies required to run the notebook.

## Data Files

The notebook expects the following Excel files to be placed in the same directory:

- `完美日记_好评.xlsx`
- `完美日记_差评.xlsx`
- `橘朵_好评.xlsx`
- `橘朵_差评.xlsx`
- `毛戈平_好评.xlsx`
- `毛戈平_差评.xlsx`
- `玛丽黛佳_好评.xlsx`
- `玛丽黛佳_差评.xlsx`

These raw data files are not included in this repository if they contain private, copyrighted, or large-scale review data.

## How To Run

Install the required packages:

```bash
pip install -r requirements.txt
