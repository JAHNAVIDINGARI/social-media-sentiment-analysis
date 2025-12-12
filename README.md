# Robust Hybrid Model for Social Media Sentiment Analysis .
<p align="justify">This project demonstrates an automated sentiment analysis system for social media text using state-of-the-art transformer models such as BERT and RoBERTa. Understanding public sentiment is critical for businesses, policymakers, and researchers to gain insights from social media data. In this project, social media text is preprocessed, tokenized, and analyzed using both individual and hybrid transformer models to classify sentiments accurately. This project was developed as part of my research portfolio, focusing on text analytics and deep learning-based sentiment classification.</p>

## Publication
<p align="justify">This project has been presented in a research context under the title: <i>"Robust Hybrid Transformer-Based Model for Social Media Sentiment Analysis"</i>. The study explores the effectiveness of hybrid BERT-RoBERTa models for sentiment classification and compares their performance against standard BERT-based models, providing insights on text representation and multi-class sentiment detection.</p>

## Citation
<p align="justify">If you use this project in your research or work, please cite as: D. Jahnavi, S. Sami, S. Pulata, G. Bharathi Mohan and R. Prasanna Kumar, "Robust Hybrid Model for Social Media Sentiment Analysis," 2024 15th International Conference on Computing Communication and Networking Technologies (ICCCNT), Kamand, India, 2024, pp. 1-6, doi: 10.1109/ICCCNT61001.2024.10725264. keywords: {Training;Performance evaluation;Analytical models;Sentiment analysis;Accuracy;Social networking (online);Computational modeling;Bidirectional control;Encoding;Context modeling;Sentiment Analysis;Machine learning;Transformers;BERT;RoBERTa;Hybrid model;Accuracy metrics;Confusion Matrix},

</p>

## Table of Contents
- [Project Overview](#project-overview)
- [Why I Chose This Project](#why-i-chose-this-project)
- [Problem This Project Solves](#problem-this-project-solves)
- [Dataset](#dataset)
- [Flow of the Project](#flow-of-the-project)
- [Files in This Repository](#files-in-this-repository)
- [Tech Stack Used and Why](#tech-stack-used-and-why)
- [Usage Instructions](#usage-instructions)
- [Results and Insights](#results-and-insights)
- [Author](#author)
- [Contact](#contact)

## Project Overview
<p align="justify">The objective of this project is to automatically detect sentiments expressed in social media text. Using transformer-based models such as BERT, RoBERTa, and a hybrid BERT-RoBERTa approach, the system classifies text into multiple sentiment categories including Positive, Excitement, Contentment, Joy, Neutral, Happy, and Miscellaneous. The workflow includes text preprocessing, tokenization, model training, evaluation, and visualization, enabling accurate sentiment analysis of social media posts.</p>

## Why I Chose This Project?
<p align="justify">Understanding public sentiment on social media has become increasingly important for businesses, policymakers, and researchers. I chose this project to address the challenge of multi-class sentiment classification and to explore hybrid transformer architectures for enhanced performance. This project helped me gain hands-on experience with NLP preprocessing, transformer tokenization, hybrid model design, and deep learning-based classification.</p>

## Problem This Project Solves
<p align="justify">Manual analysis of social media text is time-consuming and prone to error. Existing sentiment analysis tools often struggle with nuanced multi-class classification. This project provides an automated, accurate solution to detect sentiment in text, enabling organizations to extract actionable insights from social media data, understand public opinion, and make data-driven decisions.</p>

## Dataset
<p align="justify">The dataset used consists of social media posts labeled with multiple sentiment categories such as Positive, Excitement, Contentment, Joy, Neutral, Happy, and Miscellaneous. Each post contains textual content along with its corresponding sentiment label.</p> <p align="justify"><b>Dataset Link:</b> <a href="https://www.kaggle.com/datasets" target="_blank">Download Social Media Sentiment Dataset from Kaggle</a></p> <p align="justify">The data preprocessing steps include removing infrequent classes, handling miscellaneous sentiments, tokenizing text using BERT and RoBERTa tokenizers, padding sequences, and splitting the dataset into training, validation, and test sets. These steps prepare the dataset for training hybrid transformer models.</p>

## Flow of the Project
<p align="justify">The workflow of this project is designed to transform raw social media text into accurate sentiment predictions. The steps include:</p>

<b>Load Dataset</b><br>
Text data is loaded from CSV files.

<b>Data Preprocessing</b><br>

Filtering infrequent classes

Mapping sentiment labels to integers

Handling miscellaneous classes

Text length visualization

Generating word clouds for frequent terms

<b>Tokenization</b><br>

Tokenizing text using BERT and RoBERTa tokenizers

Padding sequences to uniform length

Encoding attention masks

<b>Dataset Splitting</b><br>

Splitting the dataset into training, validation, and test sets

<b>Model Building</b><br>
Transformer-based models used:

BERT

RoBERTa

Hybrid BERT-RoBERTa model

<b>Model Training</b><br>

Fine-tuning transformer models on training data

Using Adam  optimizer and learning rate scheduler

Dropout applied to prevent overfitting

<b>Evaluation</b><br>

Calculating accuracy, confusion matrix, and classification report

Comparing performance of BERT, RoBERTa, and hybrid models

<b>Prediction</b><br>

Predicting sentiment on test data

Visualizing confusion matrices for detailed analysis

## Files in This Repository

social_media_sentiment_analysis.ipynb – Jupyter Notebook containing full implementation
Research Paper & PPt – Published and presented in Prestegious conference 15th International Conference on Computing Communication and Networking Technologies (ICCCNT), Kamand, India, 2024, pp. 1-6, doi: 10.1109/ICCCNT61001.2024.10725264.
requirements.txt – List of Python dependencies
README.md – Project documentation

## Tech Stack Used and Why

Python: Core language for data analysis and model development

NumPy & Pandas: Numerical computations and data manipulation

Matplotlib & Seaborn: Visualization of text length, word clouds, and confusion matrices

NLTK: Sentiment lexicon tools and preprocessing

PyTorch & Transformers: Training and fine-tuning transformer-based models

Scikit-learn: Dataset splitting, evaluation metrics, and performance analysis

<p align="justify">These tools provide a complete ecosystem for natural language processing, deep learning model training, evaluation, and visualization of sentiment analysis results.</p>
Usage Instructions
<p align="justify">1. <b>Clone the repository</b><br> <code>git clone https://github.com/JAHNAVIDINGARI/SOCIAL-MEDIA-SENTIMENT-ANALYSIS.git</code></p> <p align="justify">2. <b>Navigate to the project directory</b><br> <code>cd social-media-sentiment-analysis</code></p> <p align="justify">3. <b>Install dependencies</b><br> <code>pip install -r requirements.txt</code></p> <p align="justify">4. <b>Run the notebook</b><br> Open <b>social_media_sentiment_analysis.ipynb</b>, update dataset path if required, and execute all cells to train and evaluate the models.</p>

## Results and Insights
<p align="justify">The hybrid BERT-RoBERTa model achieved strong performance across multi-class sentiment classification. Key observations include:</p> - BERT and RoBERTa models individually achieved high accuracy (~85–88%) but struggled with certain nuanced classes. - The hybrid model improved overall accuracy (~91%) by combining embeddings from both transformers. - Confusion matrices showed clear separation between sentiment classes, demonstrating robust multi-class sentiment detection. - Visualization of word clouds and text length distributions provided additional insights into dataset characteristics. <p align="justify">These results confirm that hybrid transformer models are effective for social media sentiment analysis and can support organizations in monitoring public opinion, brand perception, and social trends.</p>

## Authors

Dingari Jahnavi,
Sasank Sami,
Sandeep Pulata,
G Bharathi Mohan,
R Prasanna Kumar.

## Contact
<p align="justify">For queries, collaboration, or further discussion regarding this project, please reach out via <b>LinkedIn</b> or <b>email</b>:</p>

LinkedIn: https://www.linkedin.com/in/jahnavi-dingari

Email: <a href="mailto:jahnavidingari04@gmail.com">jahnavidingari04@gmail.com
</a>
