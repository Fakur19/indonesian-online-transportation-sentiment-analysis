# Competitive Landscape Analysis of Indonesian Online Transportation Apps Through Google Play Store Reviews

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-lightgrey.svg)

## Project Overview

This project presents an in-depth sentiment analysis of the leading ride-hailing applications in Indonesia: **Gojek, Grab, Maxim, and inDrive**. By leveraging a massive dataset of over 1.3 million user reviews from the Google Play Store (spanning from 2021 to 2025), this analysis aims to uncover the public's perception, identify key drivers of customer satisfaction and dissatisfaction, and map out the competitive landscape of this dynamic market.

The primary goal is to move beyond simple sentiment classification and extract actionable business insights through various analytical techniques, including time-series analysis, feature importance analysis, and aspect-based sentiment analysis (ABSA).

---

## Key Questions

This analysis seeks to answer several critical business questions:
1.  Which application holds the most positive sentiment among Android users in Indonesia?
2.  How has user sentiment for each brand evolved over the past few years?
3.  What are the main drivers of positive sentiment (strengths) and negative sentiment (weaknesses) for each application?
4.  How do the applications compare across key business aspects such as **Price, Service Quality, Driver Performance, and Application Usability**?

---

## Dataset

* **Source:** Google Play Store
* **Applications Analyzed:** Gojek, Grab, Maxim, inDrive
* **Total Reviews:** ~1.4 million
* **Time Range:** January 2021 - June 2025
* **Language:** Indonesian

---

## Methodology & Workflow

This project follows a structured, end-to-end data analysis workflow:

1.  **Data Collection:** Scraped over 1.3 million user reviews for the four target applications using the `google-play-scraper` library.
2.  **Data Preprocessing & Cleaning:** Performed extensive text cleaning, which included:
    * Lowercasing text (case folding)
    * Removing punctuation, numbers, emojis, and special characters
    * Normalizing Indonesian slang and informal words to their standard form
    * Removing common Indonesian stopwords to reduce noise.
3.  **Exploratory Data Analysis (EDA):** Investigated the dataset to uncover initial patterns through:
    * **Distribution Analysis:** Comparing review volume and time range for each app.
    * **Sentiment Distribution:** Visualizing the proportion of positive and negative reviews.
    * **Time-Series Analysis:** Analyzing trends in review volume and sentiment over time.
4.  **Sentiment Classification Modeling:**
    * Built a baseline classification model using a **TF-IDF Vectorizer** and **Logistic Regression** pipeline.
    * The model achieved a robust **accuracy of 93%**, establishing a strong benchmark for sentiment classification.
5.  **In-depth Analysis:**
    * **Feature Importance Analysis:** Extracted the most influential keywords (coefficients) from the trained model to understand what drives positive and negative sentiment for each app.
    * **Aspect-Based Sentiment Analysis (ABSA):** Categorized reviews into key business aspects (Application, Price, Service, Driver, Customer Service) to perform a granular, head-to-head comparison of each app's perceived strengths and weaknesses.

---

## Key Findings & Insights

### 1. Market Overview: Positive but Competitive
- The overall market sentiment is overwhelmingly positive (~80%), but a significant volume of negative reviews (~270,000) provides a rich source for identifying customer pain points.
- **Maxim** consistently emerges as the leader in overall positive sentiment, closely followed by **Gojek** and **Grab**.

### 2. Time-Series Trends: A Story of Consistency vs. Volatility
- **Maxim** maintains the most stable and highest sentiment rating throughout the entire 2021-2025 period.
- **Gojek and Grab** are locked in a tight race, with their sentiment trends being more volatile and frequently crossing over, likely reflecting the impact of price wars or service updates.
- **inDrive** shows the most volatility, indicating a less stable user experience, but has been on a positive recovery trend since early 2024.

### 3. Core Strengths and Weaknesses (Feature & Aspect Analysis)

| Application | Top Strength(s) | Top Weakness(es) |
| :--- | :--- | :--- |
| **Gojek** | **Functionality & Utility** (highly valued as a "helpful" and "easy-to-use" super-app) | **Application Performance** (bugs, errors) & **Customer Service** |
| **Grab** | **Service Reliability** & emotional appreciation ("best", "thanks") | **Price Perception** & **Customer Service** |
| **Maxim** | **Price (Affordability)** & **Driver Conduct** (perceived as friendly and patient) | Relatively few weaknesses, maintaining high sentiment across most aspects. |
| **inDrive** | **Price (Affordability)** due to its bargaining model | **Driver Conduct** & **Customer Service** |

### 4. The Universal Pain Point: Customer Service
Across all four platforms, **Customer Service** was identified as the aspect with the highest proportion of negative sentiment. This represents a major opportunity area for any player who can significantly improve their complaint handling and user support processes.

---

## Tools and Libraries

* **Data Manipulation & Analysis:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn, WordCloud
* **Machine Learning & NLP:** Scikit-learn (TfidfVectorizer, LogisticRegression), NLTK
* **Data Collection:** google-play-scraper
* **Development Environment:** Jupyter Notebook

---

## How to Run This Project

This project can be reproduced in two ways.

### Option 1: Using the Provided Dataset (It's Sample Dataset)
This method allows you to run the analysis notebook directly without scraping the data again.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Fakur19/indonesian-online-transportation-sentiment-analysis.git
    cd indonesian-online-transportation-sentiment-analysis
    ```
2.  **Set up a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Jupyter Notebook:**
    * Launch Jupyter: `jupyter notebook`
    * Open the main analysis notebook (e.g., `sentiment-analysis.ipynb`) and run the cells.

### Option 2: Scraping the Data from Scratch
If you wish to collect the data yourself, you can use the provided `data-scrap.py` script.

1.  **Follow steps 1-3** from Option 1 to set up the environment.
2.  **Run the scraping script:**
    ```bash
    python data-scrap.py
    ```
    *Note: This process can take a significant amount of time depending on the number of reviews and your internet connection.*
3.  **Once the data is collected**, proceed with **step 4** from Option 1 to run the analysis notebook.

---

## Future Work

- **Expand Data Sources:** Incorporate reviews from the Apple App Store to create a more comprehensive market view.
- **Advanced Modeling:** Implement and fine-tune advanced NLP models like **IndoBERT** to potentially improve classification accuracy and extract more nuanced insights.
- **Topic Modeling:** Apply techniques like Latent Dirichlet Allocation (LDA) to automatically discover hidden topics within user complaints and praises.
