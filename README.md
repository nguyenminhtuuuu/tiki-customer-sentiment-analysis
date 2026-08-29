# Tiki Customer Sentiment Analysis: Product Reviews
**Samsung Innovation Campus - Big Data Capstone Project | Team 4 Members**

**Tools & Technologies:** Python, Scikit-learn, Underthesea (NLP), TF-IDF, SVM, Logistic Regression, PhoBERT, Streamlit.


## 1. Introduction and Business Value
With the rapid growth of e-commerce, customer reviews on platforms like Tiki contain invaluable insights into consumer sentiment, experiences, and expectations. 

This project builds an automated sentiment analysis pipeline for product reviews (specifically in the Books category) to help businesses:
- Detect product quality issues early.
- Improve customer experience and optimize marketing strategies.
- Make data-driven decisions rather than relying solely on star ratings.

## 2. Dataset Overview
The data was collected from Tiki.vn. After initial cleaning, the dataset consists of approximately **88,000 review records**.

| Field Name | Description |
| :--- | :--- |
| `title` | Short review title (e.g., "Cực kỳ hài lòng") |
| `thank_count` | Number of thank-yous for comments |
| `customer_id` | Customer code (anonymous) |
| `rating` | Rating from 1 to 5 stars |
| `content` | Comment content (free text, containing customer emotions and opinions) |


## 3. Exploratory Data Analysis (EDA) - Before Training
Understanding the raw text data is crucial. We visualized the distribution of words, sentence lengths, and sentiment labels before applying any machine learning models.

### 3.1. Text Characteristics
The word cloud and frequency charts reveal that words like "sách" (book), "giao" (delivery), "nhanh" (fast), and "đẹp" (beautiful) dominate the positive reviews, highlighting that delivery speed and book quality are key satisfaction drivers.

![Top 20 Frequent Words](images/top_20_words.png)

*Figure 1: Top 20 most frequent words in the dataset.*

![Word Cloud](images/word_cloud.png)
*Figure 2: Word Cloud showing prominent keywords.*

![Sentence Length Distribution](images/sentence_length.png)
*Figure 3: Sentence length distribution.*

### 3.2. Class Imbalance Problem
An initial check on the target variable revealed a severe class imbalance. Positive reviews (4-5 stars / "Cực kỳ hài lòng") accounted for the vast majority of the dataset (76.7% for 5-star ratings). 


![Label Distribution](images/label_distribution_pie_chart.png)
*Figure 4: Label and Rating distribution highlighting extreme imbalance.*

## 4. Exploratory Data Analysis (EDA) 
To prepare the raw text for machine learning models, we built a comprehensive text processing pipeline:
- **Cleaning:** Removed duplicate records, null values, emojis, and special characters.
- **Normalization:** Converted text to lowercase and applied word segmentation using `Underthesea`.
- **Correction:** Replaced common abbreviations (e.g., "ko" → "không", "sp" → "sản phẩm") and fixed typos.

**Handling Data Imbalance:**
The original dataset was heavily skewed towards positive reviews ("cực kỳ hài lòng" accounted for ~78,000 records). We applied **Undersampling** to balance the classes and prevent model bias.


## 4. Feature Extraction & Traditional Machine Learning
We extracted features using CountVectorizer and TfidfVectorizer, then selected the most important features using SelectKBest (Chi-squared).
**Bar chart "Top 20 Frequent Words"**


We experimented with 4 traditional ML models for a 2-class problem (Cực kỳ hài lòng / Không hài lòng):
