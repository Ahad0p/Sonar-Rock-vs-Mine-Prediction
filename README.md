# 🔍 Sonar Rock vs Mine Classification - ML Project

This project is a complete machine learning pipeline to classify whether a sonar signal is bounced back from a **Rock (R)** or a **Mine (M)** using the [UCI Sonar Dataset](https://archive.ics.uci.edu/ml/datasets/connectionist+bench+sonar+mines+vs+rocks).

## 📂 Project Structure

sonar-rock-vs-mine/
│
├── artifact/ # Stores generated model, preprocessor, label encoder
├── notebook/ # Optional: Jupyter notebook explorations
├── src/
│ ├── components/
│ │ ├── data_ingestion.py # Loads and splits raw data
│ │ ├── data_transformation.py # Preprocessing, scaling, label encoding
│ │ ├── model_trainer.py # Model training, hyperparameter tuning
│ │ └── **init**.py
│ │
│ ├── pipeline/
│ │ ├── train_pipeline.py # Runs the complete training pipeline
│ │ ├── predict_pipeline.py # Loads model and predicts from new input
│ │ └── **init**.py
│ │
│ ├── exception.py # Custom error handler
│ ├── logger.py # Logging setup
│ ├── utils.py # Utility methods (save_object, load_object, evaluate_models)
│ └── **init**.py
│
├── app.py # FastAPI server
├── requirements.txt # Dependencies
├── README.md # Project guide
└── .gitignore

---

## 📦 Features

- Complete **ML pipeline**: ingestion → transformation → training → saving artifacts
- Supports **classification** (R vs M)
- Uses **label encoding**, **scaling**, and **GridSearchCV**
- Integrated with **FastAPI** for real-time predictions
- Testable via **Postman** or **Swagger UI**

---

## 🚀 Getting Started

### 1. Clone this Repo

```bash
git clone https://github.com/yourusername/sonar-rock-vs-mine.git
cd sonar-rock-vs-mine

python -m venv venv
venv\Scripts\activate      # On Windows

pip install -r requirements.txt

🧪 Train the Model
python src/pipeline/train_pipeline.py

This will:
Read data
Preprocess it
Train multiple models
Save the best one to artifact/model.pkl

⚡ Run FastAPI App
uvicorn app:app --reload
Visit: http://localhost:8000/docs

✅ Tech Stack
Python
scikit-learn,XGBoost,CatBoost
FastAPI
Pydantic
NumPy,Pandas

📌 Notes
The model is trained on a 60-feature sonar dataset
Final accuracy: ~86%
Supports easy deployment via API

👨‍💻 Author
Ahad Khan
Undergraduate, B.E AI & Data Science
Muffakhamjah College of Engineering
Connect: https://www.linkedin.com/in/mohd-abdul-ahad-khan-3b4428317 | https://github.com/Ahad0p
```
