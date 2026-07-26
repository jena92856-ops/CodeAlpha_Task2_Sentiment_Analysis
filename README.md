# Amazon Product Reviews — Sentimental Analysis 

A Python-based exploratory data analysis of an Amazon product dataset. The project cleans raw pricing, rating, and review data, then visualizes pricing trends, rating distributions, top products, review-length patterns, and customer sentiment derived from star ratings.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Analysis Walkthrough](#-analysis-walkthrough)
  - [1. Setup & Imports](#1-setup--imports)
  - [2. Loading the Data](#2-loading-the-data)
  - [3. Initial Inspection](#3-initial-inspection)
  - [4. Missing Values](#4-missing-values)
  - [5. Duplicate Removal](#5-duplicate-removal)
  - [6. Data Cleaning](#6-data-cleaning)
  - [7. Descriptive Statistics](#7-descriptive-statistics)
  - [8. Top Product Categories](#8-top-product-categories)
  - [9. Rating Distribution](#9-rating-distribution)
  - [10. Discount Percentage Distribution](#10-discount-percentage-distribution)
  - [11. Top Rated Products](#11-top-rated-products)
  - [12. Most Reviewed Products](#12-most-reviewed-products)
  - [13. Actual vs Discounted Price](#13-actual-vs-discounted-price)
  - [14. Correlation Heatmap](#14-correlation-heatmap)
  - [15. Review Length Distribution](#15-review-length-distribution)
  - [16. Sentiment Labeling](#16-sentiment-labeling)
  - [17. Sentiment Distribution](#17-sentiment-distribution)
  - [18. Average Rating by Sentiment](#18-average-rating-by-sentiment)
- [Key Findings](#-key-findings)
- [Graphs Generated](#-graphs-generated)
- [License](#-license)

---

## 📖 Overview

This notebook (`Task_2.ipynb`) performs an end-to-end exploratory data analysis on an Amazon products dataset (`amazon.csv.zip`). It covers:

- Data loading and inspection
- Cleaning currency-formatted price columns, comma-formatted counts, and percentage strings
- Handling missing values and duplicate rows
- Visualizing category distribution, rating distribution, discount distribution
- Identifying top-rated and most-reviewed products
- Studying the relationship between actual price, discounted price, rating, and rating count
- Deriving a simple sentiment label (`Positive` / `Neutral` / `Negative`) from numeric ratings and visualizing its distribution

---

## 📊 Dataset

The dataset used is `amazon.csv.zip`, containing Amazon product listings with the following columns:

| Column | Description |
|---|---|
| `product_id` | Unique identifier for the product |
| `product_name` | Name/title of the product |
| `category` | Pipe-separated (`|`) category hierarchy |
| `discounted_price` | Sale price (originally formatted with `₹` and commas) |
| `actual_price` | Original/list price (originally formatted with `₹` and commas) |
| `discount_percentage` | Discount applied (originally formatted as a `%` string) |
| `rating` | Average customer star rating |
| `rating_count` | Number of ratings received (originally comma-formatted) |
| `about_product` | Bullet-style product description |
| `user_id` | Comma-separated list of reviewer IDs |
| `user_name` | Comma-separated list of reviewer names |
| `review_id` | Comma-separated list of review IDs |
| `review_title` | Comma-separated list of review titles |
| `review_content` | Comma-separated list of review text |
| `img_link` | Product image URL |
| `product_link` | Amazon product page URL |

> **Note:** The raw CSV is not included in this repository. Place `amazon.csv.zip` in a `/content` (Colab) or local `data/` folder and update the file path in the notebook accordingly before running.

---

## 🛠 Tech Stack

- **Python 3**
- **pandas** — data loading, cleaning, and manipulation
- **numpy** — numerical operations
- **matplotlib** — base plotting
- **seaborn** — statistical visualizations
- Developed and tested in **Google Colab**

---

## 📁 Project Structure

```
.
├── Task_2.ipynb        # Main analysis notebook
├── data/
│   └── amazon.csv.zip  # Raw dataset (not tracked in repo)
└── README.md            # Project documentation
```

---

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn jupyter
   ```

4. Add the dataset (`amazon.csv.zip`) to your working directory or update the file path used in the notebook.

---

## ▶️ Usage

1. Launch Jupyter Notebook or open the file in Google Colab:
   ```bash
   jupyter notebook Task_2.ipynb
   ```

2. Update the dataset path if needed:
   ```python
   file = pd.read_csv('/content/amazon.csv.zip')
   # or, for a local copy:
   # file = pd.read_csv('data/amazon.csv.zip')
   ```

3. Run all cells sequentially (`Runtime > Run all` in Colab, or `Kernel > Restart & Run All` in Jupyter).

---

## 🔍 Analysis Walkthrough

### 1. Setup & Imports

Core libraries are imported and the seaborn plotting style is set for consistent, clean visuals across the notebook.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
sns.set_style("whitegrid")
```

### 2. Loading the Data

The dataset is read directly from a zipped CSV file into a pandas DataFrame, and the first five rows are previewed.

```python
import pandas as pd
file = pd.read_csv('/content/amazon.csv.zip')
file.head()
```

### 3. Initial Inspection

Basic structural checks are performed: dataset shape, column names, and data types/non-null counts.

```python
file.shape
```

```python
file.columns
```

```python
file.info()
```

```python
file.describe(include='all')
```

### 4. Missing Values

Missing values are counted per column and visualized as a bar chart to quickly spot problem columns.

```python
file.isnull().sum()
```

```python
plt.figure(figsize=(12, 6))
file.isnull().sum().plot(kind='bar')
plt.title('Missing Values Per Column')
plt.xlabel('Columns')
plt.ylabel('Number of Missing Values')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

📈![image alt](https://github.com/jena92856-ops/CodeAlpha_Task2_Sentiment_Analysis/blob/9d6af6c1d5f5be65c794fee8758c4b143a892f44/01_missing_values.png) 

### 5. Duplicate Removal

Duplicate rows are identified and dropped to ensure clean, unique records for analysis.

```python
file.duplicated().sum()
```

```python
file = file.drop_duplicates()
```

### 6. Data Cleaning

Price columns (`discounted_price`, `actual_price`) have their `₹` currency symbol and thousands-separator commas stripped, then are cast to numeric types. The `rating` and `rating_count` columns are similarly converted to numeric, coercing any invalid entries to `NaN`.

```python
file['discounted_price'] = file['discounted_price'].astype(str).str.replace('₹','',regex=False)
file['discounted_price'] = file['discounted_price'].astype(str).str.replace(',','',regex=False)

file['actual_price'] = file['actual_price'].astype(str).str.replace('₹','',regex=False)
file['actual_price'] = file['actual_price'].astype(str).str.replace(',','',regex=False)

file['discounted_price'] = pd.to_numeric(file['discounted_price'])
file['actual_price'] = pd.to_numeric(file['actual_price'])
```

```python
file['rating'] = pd.to_numeric(file['rating'], errors='coerce')
```

```python
file['rating_count'] = file['rating_count'].astype(str).str.replace(',','')
file['rating_count'] = pd.to_numeric(file['rating_count'], errors='coerce')
```

### 7. Descriptive Statistics

Summary statistics (mean, std, min, max, quartiles) are generated for the key numeric fields.

```python
file[['discounted_price','actual_price','rating','rating_count']].describe()
```

### 8. Top Product Categories

The 10 most frequent product categories are extracted (using only the most specific/last segment of the pipe-separated category path) and plotted as a bar chart.

```python
# Get the top 10 categories and their counts
top_categories_data = file['category'].value_counts().head(10)

# Extract the last subcategory for better readability
short_categories_labels = top_categories_data.index.map(lambda x: x.split('|')[-1])

plt.figure(figsize=(14,8))
sns.barplot(x=short_categories_labels, y=top_categories_data.values, hue=short_categories_labels, palette='viridis', legend=False)

plt.title("Top 10 Product Categories", fontsize=16)
plt.xlabel("Category", fontsize=12)
plt.ylabel("Number of Products", fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()
```

📈![image alt](https://github.com/jena92856-ops/CodeAlpha_Task2_Sentiment_Analysis/blob/a18ffe5813ce8521f132709def33a10b6ad28b64/02_top_categories.png) 

### 9. Rating Distribution

A histogram (with KDE overlay) shows how customer star ratings are distributed across the dataset.

```python
plt.figure(figsize=(10,6))
sns.histplot(file['rating'], bins=10, kde=True, color='skyblue')

plt.title("Product Rating Distribution", fontsize=16)
plt.xlabel("Rating", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
```

📈![image alt](https://github.com/jena92856-ops/CodeAlpha_Task2_Sentiment_Analysis/blob/7faa045a70973854f9eb523066554107a426a7f6/03_rating_distribution.png) 

### 10. Discount Percentage Distribution

The `discount_percentage` column is cleaned (stripping `%` and converting to numeric), then its distribution is visualized.

```python
# Clean the 'discount_percentage' column first
file['discount_percentage'] = file['discount_percentage'].astype(str).str.replace('%', '', regex=False)
file['discount_percentage'] = pd.to_numeric(file['discount_percentage'], errors='coerce')

plt.figure(figsize=(10,6))
sns.histplot(file['discount_percentage'], bins=30, kde=True, color='lightcoral')

plt.title("Discount Percentage Distribution", fontsize=16)
plt.xlabel("Discount Percentage (%)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
```

📈![image alt](https://github.com/jena92856-ops/CodeAlpha_Task2_Sentiment_Analysis/blob/78bb8f621fbe55eb97c52ebc826b3768d667e177/04_discount_distribution.png)
### 11. Top Rated Products

The dataset is sorted by rating (descending) and the top 10 highest-rated products are displayed as a horizontal bar chart.

```python
top_rating = file.sort_values(by='rating', ascending=False)

top_rating[['product_name','rating']].head(10)
plt.figure(figsize=(10,6))

sns.barplot(x='rating', y='product_name', data=top_rating.head(10))

plt.title("Top Rated Products")
plt.show()
```

📈![image alt](https://github.com/jena92856-ops/CodeAlpha_Task2_Sentiment_Analysis/blob/78bb8f621fbe55eb97c52ebc826b3768d667e177/05_top_rated_products.png) 
### 12. Most Reviewed Products

The dataset is sorted by `rating_count` (descending) to find the products with the most customer engagement, plotted as a horizontal bar chart.

```python
top_reviews = file.sort_values(by='rating_count', ascending=False)

top_reviews[['product_name','rating_count']].head(10)
plt.figure(figsize=(10,6))

sns.barplot(x='rating_count', y='product_name', data=top_reviews.head(10))

plt.title("Most Reviewed Products")
plt.show()
```

📈![image alt](https://github.com/jena92856-ops/CodeAlpha_Task2_Sentiment_Analysis/blob/f8737ff13a28c6ad632392d32b486638a5b98aeb/06_most_reviewed_products.png)

### 13. Actual vs Discounted Price

A scatter plot explores the relationship between a product's original price and its discounted sale price.

```python
plt.figure(figsize=(8,6))

sns.scatterplot(x=file['actual_price'], y=file['discounted_price'])

plt.title("Actual Price vs Discounted Price")
plt.show()
```

📈![image alt](https://github.com/jena92856-ops/CodeAlpha_Task2_Sentiment_Analysis/blob/8be4f83aa25b94618348ff8ad4fe7d7b28e26110/07_price_comparison_scatter.png)

### 14. Correlation Heatmap

A correlation matrix is computed across the key numeric fields (`actual_price`, `discounted_price`, `rating`, `rating_count`) and rendered as an annotated heatmap.

```python
plt.figure(figsize=(8,6))

sns.heatmap(file[['actual_price','discounted_price','rating','rating_count']].corr(),
            annot=True, cmap='coolwarm')

plt.title("Correlation Heatmap")
plt.show()
```

📈 **Graph:** *Correlation Heatmap* — annotated heatmap of pairwise correlations between price, rating, and rating count.

### 15. Review Length Distribution

A new `review_length` feature is engineered (character count of `review_content`), then its distribution is visualized.

```python
file['review_length'] = file['review_content'].astype(str).apply(len)

file['review_length'].describe()
plt.figure(figsize=(8,5))

sns.histplot(file['review_length'], bins=30, kde=True)

plt.title("Review Length Distribution")
plt.show()
```

📈 **Graph:** *Review Length Distribution* — histogram with KDE curve of review text lengths.

### 16. Sentiment Labeling

A simple rule-based sentiment label is derived from the numeric `rating`:

- `rating >= 4` → **Positive**
- `3 <= rating < 4` → **Neutral**
- `rating < 3` → **Negative**

```python
def sentiment(x):
    if x >= 4:
        return "Positive"
    elif x >= 3:
        return "Neutral"
    else:
        return "Negative"

file['Sentiment'] = file['rating'].apply(sentiment)
file['Sentiment'].value_counts()
```

### 17. Sentiment Distribution

A count plot shows how many products fall into each sentiment category.

```python
plt.figure(figsize=(6,5))

sns.countplot(x='Sentiment', data=file)

plt.title("Sentiment Distribution")
plt.show()
```

📈 **Graph:** *Sentiment Distribution* — count plot of Positive / Neutral / Negative product counts.

### 18. Average Rating by Sentiment

The mean rating is computed per sentiment class and plotted as a bar chart, serving as a sanity check on the labeling logic.

```python
file.groupby('Sentiment')['rating'].mean()
sns.barplot(x='Sentiment', y='rating', data=file)

plt.show()
```

📈 **Graph:** *Average Rating by Sentiment* — bar chart of mean rating per sentiment category.

---

## 🔑 Key Findings

- Most products have high ratings (4–5 stars), indicating generally positive customer satisfaction.
- Electronics and accessories dominate the dataset.
- Discounted prices are consistently lower than actual prices, reflecting promotional pricing.
- Products with higher ratings often receive more reviews, suggesting popularity and customer trust.
- Review lengths vary widely, indicating differences in customer engagement.
- The sentiment distribution is heavily skewed toward Positive, with fewer Neutral and Negative reviews.

---

## 📉 Graphs Generated

| # | Chart | Type |
|---|---|---|
| 1 | Missing Values Per Column | Bar chart |
| 2 | Top 10 Product Categories | Bar chart |
| 3 | Product Rating Distribution | Histogram + KDE |
| 4 | Discount Percentage Distribution | Histogram + KDE |
| 5 | Top Rated Products | Horizontal bar chart |
| 6 | Most Reviewed Products | Horizontal bar chart |
| 7 | Actual Price vs Discounted Price | Scatter plot |
| 8 | Correlation Heatmap | Heatmap |
| 9 | Review Length Distribution | Histogram + KDE |
| 10 | Sentiment Distribution | Count plot |
| 11 | Average Rating by Sentiment | Bar chart |

---

## 📄 License

This project is provided for educational purposes. Add a license of your choice (e.g., MIT) if you plan to distribute this repository publicly.
