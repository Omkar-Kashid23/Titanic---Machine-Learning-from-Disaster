# Titanic Survival Prediction

## 📖 Project Overview
This project aims to predict whether a passenger survived the Titanic disaster using machine learning techniques. It involves data exploration, feature engineering, preprocessing, and modeling using a Random Forest Classifier. The goal is to achieve high accuracy on the test dataset and generate a submission file compatible with Kaggle competitions.

## 📂 Dataset
The project uses the classic **Titanic Dataset** (typically available on Kaggle), consisting of two main files:
- `train.csv`: Training data containing passenger details and survival status (Target: `Survived`).
- `test.csv`: Test data containing passenger details without survival status (used for final predictions).

**Key Features:**
- `PassengerId`: Unique identifier
- `Survived`: Target variable (0 = No, 1 = Yes)
- `Pclass`: Ticket class (1, 2, 3)
- `Name`: Passenger name
- `Sex`: Gender
- `Age`: Age in years
- `SibSp`: Number of siblings/spouses aboard
- `Parch`: Number of parents/children aboard
- `Ticket`: Ticket number
- `Fare`: Passenger fare
- `Cabin`: Cabin number
- `Embarked`: Port of embarkation (C, Q, S)

## 🛠️ Requirements
To run this project, ensure you have the following Python libraries installed:

```bash
pip pandas numpy matplotlib seaborn scikit-learn
```
## 🚀 Methodology

### 1. Data Loading & Exploration
- Loaded training and testing datasets.
- Performed initial inspection using `head()`, `info()`, and `describe()`.
- Identified missing values in `Age`, `Cabin`, `Embarked`, and `Fare` (test set).

### 2. Data Preprocessing & Feature Engineering
Several transformations were applied to improve model performance:

- **Missing Value Imputation:**
  - `Age`: Filled with mean age from the training set.
  - `Embarked`: Filled with mode ('S').
  - `Fare` (Test set): Filled with median fare from the training set.
  - *Note: Training statistics were used to fill test data to prevent data leakage.*

- **Feature Extraction:**
  - **Deck:** Extracted the first letter of the `Cabin` column. Missing cabins were marked as 'U'. Rare decks ('T', 'G', 'A') were grouped into 'Other'.
  - **Title:** Extracted titles (Mr, Mrs, Miss, Master, etc.) from the `Name` column using Regex. Rare titles were grouped into 'Rare'.
  - **Group Size:** Calculated based on the frequency of the `Ticket` number.
  - **Real Fare:** Calculated as `Fare / Group_Size` to normalize fare per person.

- **Encoding:**
  - **Sex:** Mapped to binary (female: 0, male: 1).
  - **Embarked:** Mapped to integers (S: 0, C: 1, Q: 2).
  - **Deck:** Label Encoded.
  - **Title:** One-Hot Encoded (created columns `T_0`, `T_1`, etc.).

- **Dropped Columns:**
  - `PassengerId`, `Name`, `Ticket`, `Cabin`, `Fare` (removed after engineering new features).

### 3. Modeling
- **Algorithm:** Random Forest Classifier
- **Hyperparameters:**
  - `n_estimators`: 250
  - `max_depth`: 5
  - `random_state`: 42
- **Validation:** Data was split into training and validation sets (80/20) to evaluate performance before final submission.

## 📊 Results
The model achieved the following performance metrics on the validation set:

| Metric | Score |
| :--- | :--- |
| **Accuracy** | ~81% |
| **Precision (Class 0)** | 0.81 |
| **Recall (Class 0)** | 0.88 |
| **Precision (Class 1)** | 0.80 |
| **Recall (Class 1)** | 0.72 |

## 📁 Project Structure
```
├── titanic/
│   ├── train.csv          # Training data
│   └── test.csv           # Test data
├── Titanic_Survival_Prediction.ipynb  # Main notebook
├── submission.csv         # Generated predictions for Kaggle
└── README.md              # This file
```

## 💻 Usage

1. **Clone or Download** this repository.
2. **Prepare Data:** Ensure `train.csv` and `test.csv` are located in a folder named `New folder` (or update the path in the notebook).
3. **Run the Notebook:**
   Open `Titanic_Survival_Prediction.ipynb` in Jupyter Notebook or Jupyter Lab.
   Run all cells sequentially.
4. **Output:**
   The final cell will generate a `submission.csv` file containing `PassengerId` and predicted `Survived` status.

## 🔍 Key Code Snippets

**Feature Engineering (Title Extraction):**
```python
train['Title'] = train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
train['Title'] = train['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
```

**Model Training:**
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=250, max_depth=5, random_state=42)
model.fit(x_train, y_train)
```

## 🤝 Future Improvements
- Experiment with other algorithms (e.g., XGBoost, SVM, Logistic Regression).
- Perform hyperparameter tuning using `GridSearchCV`.
- Explore additional feature interactions (e.g., Family Size vs. Survival).
- Handle outliers in `Fare` and `Age` more robustly.

## 📄 License
This project is available for only educational purposes.
```
