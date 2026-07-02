# Marketing Channel ROI Prediction & Budget Optimization

A machine learning project that predicts whether a newly acquired customer is likely to become a **high-value customer** and uses those predictions to recommend how marketing budgets should be distributed across acquisition channels for better ROI.

Instead of only focusing on prediction accuracy, this project connects machine learning with business decision-making. It combines customer behavior analysis, predictive modeling, explainable AI, and budget optimization to help answer a practical business question:

> **"If the marketing budget stays the same, where should we spend it to generate the highest return?"**

The project is built using real-world data from the **Olist Brazilian E-commerce Dataset**, with a small, clearly documented CAC benchmark overlay because customer acquisition costs are not publicly available.

---

# Dataset

This project uses the **Olist Marketing Funnel** and **Brazilian E-commerce** datasets.

Each record represents an acquired seller, starting from a marketing-qualified lead (MQL) that eventually became a customer. These datasets allow us to connect:

* Acquisition channel
* Business information
* Purchase behavior
* Revenue generated

The only feature that is not directly available in the dataset is **Customer Acquisition Cost (CAC)**. Since Olist does not publish advertising costs, reasonable channel-specific benchmark values are used. This assumption is documented throughout the project.

The final dataset contains:

* 842 acquired sellers
* 9 acquisition channels
* Around 28% high-value customers

This naturally creates an imbalanced classification problem similar to real business scenarios.

---

# Business Problem

Different marketing channels bring customers with different acquisition costs and lifetime values.

For example:

* Email campaigns are usually inexpensive but often attract loyal customers.
* Paid advertising can generate high volumes but may also have much higher acquisition costs.

The objective is to identify customers who are likely to become valuable early in their journey and recommend how marketing budgets should be redistributed toward the most profitable channels.

---

# Project Workflow

The project follows a complete machine learning pipeline:

### 1. Exploratory Data Analysis

* Analyze acquisition channels
* Compare revenue across channels
* Calculate ROI
* Study customer value distribution

### 2. Feature Engineering

Features such as:

* First order value
* Days to second purchase
* Value per acquisition cost
* Signup month
* Signup weekday
* Fast repeat purchase indicator

Categorical variables are one-hot encoded, while numerical features are standardized using a Scikit-learn `ColumnTransformer`.

To avoid data leakage, **total revenue is never used as a model feature**.

### 3. Model Training

Three models are trained and compared:

* Logistic Regression
* Random Forest
* XGBoost

Performance is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

The models are trained using stratified cross-validation, and class imbalance is handled through model-specific weighting.

SHAP is used to explain feature importance and improve model transparency.

### 4. Business Optimization

Predicted probabilities are aggregated at the acquisition channel level.

The project then estimates:

* Expected customer quality
* Expected revenue
* ROI per channel

Finally, it recommends how to reallocate a fixed marketing budget to maximize expected returns.

---

# Results

## Model Performance

| Model               | Accuracy  | ROC-AUC   |
| ------------------- | --------- | --------- |
| Random Forest       | **89.6%** | **0.952** |
| Logistic Regression | 90.1%     | 0.951     |
| XGBoost             | 88.6%     | 0.951     |

Random Forest achieved the best overall performance while maintaining excellent recall for identifying future high-value customers.

---

## Channel Insights

The analysis showed that:

* **Email**, **Referral**, and **Direct** generated the strongest ROI because of their relatively low acquisition costs.
* **Organic Search** consistently performed well and should continue receiving investment.
* **Paid Search**, **Social**, and **Display** had significantly higher acquisition costs, making them less efficient despite attracting reasonable customer quality.

---

## Budget Optimization

Using a simulated marketing budget of **$250,000**, the optimized allocation achieved:

* **~80% more customers acquired**
* **~74% higher expected revenue**
* **~90% improvement in ROI**

Rather than increasing spending, the improvement comes from shifting budget toward channels with higher expected returns.

---

# Key Business Recommendations

* Increase investment in **Email**, **Referral**, and **Direct** channels.
* Maintain investment in **Organic Search**.
* Optimize or reduce spending on **Paid Search**, **Social**, and **Display**.
* Improve attribution tracking for the "Unknown" channel.
* Re-run the model regularly since customer behavior and channel performance change over time.

---

# Project Structure

```text
channel_roi_ml/
├── main.py
├── requirements.txt
├── data/
├── models/
├── outputs/
└── src/
    ├── config.py
    ├── real_data.py
    ├── preprocessing.py
    ├── modeling.py
    ├── business_layer.py
    ├── visualization.py
    └── eda.py
```

---

# Running the Project

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt

python main.py
```

The project automatically downloads and caches the required Olist datasets if they are not already available.

---

# Visualizations

The project generates several visualizations, including:

* Channel-wise CAC
* Revenue by channel
* ROI comparison
* High-value customer distribution
* Confusion Matrix
* ROC Curves
* Feature Importance
* SHAP Summary Plot
* Budget Allocation (Before vs After)
* Revenue and ROI Simulation

---

# Highlights

* Uses real commercial data from the Olist dataset
* End-to-end machine learning pipeline
* No data leakage
* Handles class imbalance correctly
* SHAP-based model explainability
* Business-focused budget optimization
* Fully reproducible workflow
* Production-ready preprocessing pipeline

---

# Future Improvements

Some possible extensions include:

* Using real advertising spend instead of benchmark CAC values
* Hyperparameter tuning with GridSearchCV
* Adding optimization under diminishing returns using `scipy.optimize`
* Deploying the trained model with FastAPI for real-time predictions

---

**Data Source:** Olist Brazilian E-commerce Public Dataset and Marketing Funnel Dataset (publicly released and anonymized). Customer Acquisition Cost (CAC) values are benchmark estimates added for business simulation purposes.
