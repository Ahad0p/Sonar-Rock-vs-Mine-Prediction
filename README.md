# 🔍 Sonar Rock vs Mine Classification - ML Project

This project is a complete machine learning pipeline to classify whether a sonar signal is bounced back from a **Rock (R)** or a **Mine (M)** using the [UCI Sonar Dataset](https://archive.ics.uci.edu/ml/datasets/connectionist+bench+sonar+mines+vs+rocks).

## 📂 Project Structure

<pre> ```text sonar-rock-vs-mine/ ├── artifact/ ├── notebook/ ├── src/ │ ├── components/ │ │ ├── data_ingestion.py │ │ ├── data_transformation.py │ │ ├── model_trainer.py │ │ └── __init__.py │ ├── pipeline/ │ │ ├── train_pipeline.py │ │ ├── predict_pipeline.py │ │ └── __init__.py │ ├── exception.py │ ├── logger.py │ ├── utils.py │ └── __init__.py ├── app.py ├── requirements.txt ├── README.md └── .gitignore ``` </pre>

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
