# HackerEarth Machine Learning Challenge: World Water Day

## 🌍 Challenge Overview
Water scarcity is a growing global concern, and urban households contribute significantly to wastage due to inefficient consumption habits. Traditional water meters provide only total usage data, making it difficult for homeowners to optimize their water usage. This challenge aims to build a **smart water monitoring system** that predicts household water consumption patterns using machine learning.

## 🚀 Task
Your goal is to develop a **Machine Learning model** that predicts **daily water consumption** for individual households based on historical data, household demographics, weather conditions, and conservation behaviors.

## 📊 Dataset Description
The dataset contains the following files:
- **train.csv** (14000 x 12)
- **test.csv** (6000 x 11)
- **sample_submission.csv** (5 x 2)

### 📌 Columns in the dataset:
| Column Name                 | Description |
|-----------------------------|-------------|
| Timestamp                   | Unique timestamp of an entry |
| Residents                   | Number of people living in the household |
| Apartment_Type              | Type of apartment |
| Temperature                 | Average temperature at that time period |
| Humidity                    | Average humidity at that time period |
| Water_Price                 | Water price at that time period |
| Period_Consumption_Index    | Relative water usage per 8-hour period |
| Income_Level                | Household income level |
| Guests                      | Number of guests |
| Amenities                   | Types of amenities available |
| Appliance_Usage             | Whether water appliances are in use or not |
| Water_Consumption (target)  | Water consumption in that period |

----

## Folder Structure

WaterConsumptionPrediction/
|--- artifacts/
|   |-- model.pkl
|   |-- preprocessor.pkl
|   |-- standard_scaler.pkl
|   |-- train.csv
|   |-- test.csv
|   |-- predicted_data.csv
|   |-- submission.csv
|   |-- train_performance.csv
|   |-- test_performance.csv
|___ dataset/
|    |-- sample_submission.csv
|    |-- test.csv
|    |-- train.csv
│── src/
|   |--- components/
|   |    |--- data_ingestion.py
|   |    |--- data_transformation.py
|   |    |--- model_training.py
|   ├── predict_pipeline/
|   │   ├── prediction.py
|   ├── exception.py
|   |__ logger.py
|   |__ utils.py
│── app.py
│── requirements.txt
│── README.md
│── setup.py
│── submission.ipynb

----

## 📈 Evaluation Metric
The model is evaluated using the **Root Mean Squared Error (RMSE)** metric:

\[ \text{score} = \max(0, 100 - \sqrt{\text{MSE}}) \]

---

## 🔍 Steps Taken to Solve the Problem  

### 1️⃣ **Data Exploration & Cleaning**
- Loaded and inspected the dataset (`train.csv` and `test.csv`).
- Checked for missing values and handled them appropriately.
- Converted categorical variables (`Apartment_Type`, `Income_Level`) into numerical values.
- Extracted date-time features from `Timestamp` (hour, day, month, etc.).
- Normalized numerical features (`Temperature`, `Humidity`, etc.).

### 2️⃣ **Feature Engineering**
- Created new time-based features (e.g., `hour_of_day`, `day_of_week`).
- Engineered interaction terms (e.g., `Residents * Guests`, `Temperature * Humidity`).
- Encoded categorical variables using **One-Hot Encoding**.
- Performed feature selection to remove low-importance variables.

### 3️⃣ **Model Selection & Training**
- Tested multiple regression models:
  - **Baseline:** Linear Regression, SVR
  - **Tree-based models:** Decision Tree, Random Forest, XGBoost, Gradient Boost, Adaboost
- Optimized hyperparameters using **GridSearchCV**.
- Used **cross-validation** to avoid overfitting.
- Selected **GradientBoost** as the best-performing model based on **RMSE score**.

### 4️⃣ **Evaluation & Results**
- Split the dataset into **training (80%)** and **validation (20%)** sets.
- Used **RMSE (Root Mean Squared Error)** as the performance metric.
- Compared model performance against baseline approaches.
- Fine-tuned the model for better generalization.

### 5️⃣ **Predictions & Submission**
- Used the final **XGBoost model** to generate predictions on `test.csv`.
- Ensured predictions matched the required submission format.
- Saved predictions to `submission.csv` and submitted.

---

## 🛠️ Tech Stack & Libraries  
- **Programming Language:** Python 🐍  
- **Libraries Used:**  
  - `pandas`, `numpy` → Data preprocessing  
  - `matplotlib`, `seaborn` → Data visualization  
  - `sklearn` → Machine Learning models  
  - `xgboost`, `GradientBoost` → Advanced regression models   

---

## 📈 Results & Insights  
- **GradientBoost outperformed other models**, achieving the best RMSE score.  
- **Feature importance analysis** showed that `Residents`, `Period_Consumption_Index`, and `Temperature` had the highest impact on water usage.  

---

## 🚀 How to Run the Project  

### 1️⃣ **Clone the Repository**  
```bash
git clone https://github.com/your-username/World_Water_Day_Prediction.git
cd World_Water_Day_Prediction

