# Human Development Index (HDI) Predictor

An interactive, end-to-end Machine Learning web application designed to predict the **Human Development Index (HDI)** of countries. The project leverages **Python**, **Flask**, **Scikit-Learn**, and **Seaborn** to build a predictive engine and data analysis dashboard, presenting results through a responsive, modern dark-themed multi-page web interface.

---

## 🚀 Key Features

*   **Multi-Page Architecture**: Realigned with standard project guidelines into distinct, dedicated pages:
    *   **Home Page (`home.html`)**: Features an introduction to the Human Development Index (HDI) and its three dimensions with a quick-launch link to the engine.
    *   **Prediction Page (`index.html`)**: Allows users to select a country from an alphabetical dropdown, adjust key indicators, and submit form posts to calculate predictions.
*   **Predictive ML Engine**: Uses a trained **Linear Regression** model with 5 variables (including label-encoded country codes) to predict a country's HDI score, classify it into the official UNDP tiers (**Very High, High, Medium, or Low**), calculate dimension-specific indices, and recommended comparable countries.
*   **10-Plot Insights Hub**: Expanded Exploratory Data Analysis (EDA) gallery featuring exactly 10 statistical plots (strip plots, distributions, heatmaps, and scatter plots) rendered using the first 20 rows of the dataset to ensure a clean, uncluttered visual layout.
*   **Label Encoding Integration**: Features dynamic label encoding for country categories, allowing spatial representations to serve as active model parameters.

---

## 🛠️ Technology Stack

*   **Backend Web Framework**: `Flask` (Python web server handling routing and HTML form POST requests)
*   **Machine Learning**: `Scikit-Learn` (Linear Regression model fitting & metrics evaluation)
*   **Data Manipulation**: `Pandas` & `NumPy` (Feature engineering, cleaning, label encoding, and indexing)
*   **Data Visualization**: `Seaborn` & `Matplotlib` (Exploratory Data Analysis plots generation)
*   **Serialization**: `Pickle` (Saves both trained ML model and country `LabelEncoder` parameters)
*   **Frontend Interface**: `HTML5`, `Vanilla CSS3` (Premium space-dark design system, sliding transitions, glowing accents, and responsive layout).

---

## 📂 Codebase File Structure

```
SkillWallet/ (Project Directory)
├── app.py                  # Flask web backend application hosting routes (GET/POST)
├── download_data.py        # Data pipeline: cleans, normalizes, label encodes, and sorts columns
├── eda.py                  # Generates exactly 10 statistical visualizations from first 20 rows
├── train_model.py          # Splits dataset, trains model by index numbers, and serializes it
├── hdi_model.pkl           # Serialized Linear Regression model
├── country_encoder.pkl     # Serialized LabelEncoder for country categories
├── hdi_processed.csv       # Clean preprocessed country dataset
├── hdi_raw.csv             # Raw Kaggle-downloaded CSV file
├── model_summary.txt       # Saved coefficients and testing performance logs
├── requirements.txt        # Required Python libraries list
├── .gitignore              # Ignores pycache, IDE configs, and environments
├── templates/
│   ├── home.html           # Landing page with introduction and portal link
│   ├── index.html          # Prediction form and report template
│   └── insights.html       # Visual insights dashboard with lightbox zoom modal
└── static/
    ├── css/
    │   └── style.css       # Premium space-dark stylesheet
    └── plots/              # Pre-rendered 10 Exploratory Data Analysis charts
        ├── correlation_matrix.png
        ├── education_vs_hdi.png
        ├── expected_schooling_vs_hdi.png
        ├── gni_vs_hdi.png
        ├── hdi_distribution.png
        ├── life_expectancy_distribution.png
        ├── life_expectancy_vs_hdi.png
        ├── mean_schooling_distribution.png
        ├── schooling_vs_hdi_strip.png
        └── strip_plot.png
```

---

## ⚙️ Installation & Setup Instructions

Follow these step-by-step instructions to get the application running on your local machine:

### 1. Prerequisites
Make sure you have **Python 3.10 or higher** installed on your system. You can check your version by running:
```bash
python --version
```

### 2. Clone/Open the Project Directory
Navigate into the project root directory in your terminal:
```bash
cd d:/Projects/SkillWallet-HDI
```

### 3. Create a Virtual Environment (Recommended)
Isolate the project dependencies by creating a python virtual environment:
*   **Windows (PowerShell)**:
    ```powershell
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    ```
*   **macOS / Linux**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

### 4. Install Dependencies
Install all the required data science and web libraries listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 5. Run the Data Pipeline (Optional)
The dataset, plots, and serialized model are checked into Git. However, you can run the files sequentially if you want to rebuild the model and charts from scratch:
*   **Download & Preprocess Data**:
    ```bash
    python download_data.py
    ```
*   **Generate EDA Visualizations**:
    ```bash
    python eda.py
    ```
*   **Train and Serialize Model**:
    ```bash
    python train_model.py
    ```

### 6. Start the Web Server
Launch the Flask development server:
```bash
python app.py
```

Once running, open your web browser and navigate to:
*   👉 Home Page: **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)**
*   👉 Prediction Engine Page: **[http://127.0.0.1:5000/predict](http://127.0.0.1:5000/predict)**
*   👉 Visual Insights Page: **[http://127.0.0.1:5000/insights](http://127.0.0.1:5000/insights)**
