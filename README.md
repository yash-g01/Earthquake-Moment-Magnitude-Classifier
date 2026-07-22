# 🌍 Earthquake Moment Magnitude Classifier

A Machine Learning-powered web application built with **Streamlit** and **Python** designed to classify earthquake moment magnitudes ($M_w$) based on seismic and geophysical parameters.

🚀 **Live Demo:** [Earthquake Moment Magnitude Classifier on Streamlit](https://earthquake-moment-magnitude-classifier-mvkbcyyahegwfedqxkzdfv.streamlit.app/)

---

## 📌 Overview

Earthquake magnitude determination is crucial for seismic hazard assessment, emergency response, and earthquake engineering. The **Earthquake Moment Magnitude Classifier** uses machine learning techniques to process seismic attributes and categorize earthquake moment magnitudes efficiently.

This repository contains:
- **`app.py`**: An interactive web app built using Streamlit for real-time model inference and user interactions.
- **`classify.ipynb`**: A comprehensive Jupyter Notebook covering data cleaning, exploratory data analysis (EDA), model training, and performance evaluation.
- **`earthquake_classifier_model.pkl`**: A serialized, pre-trained scikit-learn classifier ready for deployment.
- **`.devcontainer/`**: Configuration for automated, reproducible development environments with VS Code / GitHub Codespaces. This is for streamlit app.

---

## ✨ Features

- ⚡ **Real-time Classification:** Input seismic parameters and obtain instant moment magnitude classifications.
- 🎨 **User-Friendly Interface:** Streamlit-powered interactive UI with responsive controls.
- 📓 **Reproducible ML Workflow:** Well-documented Jupyter notebook tracing feature engineering, training, and metrics.
- 🐳 **Devcontainer Integration:** Pre-configured Docker-based dev environment for seamless setup.

---

## 📂 Repository Structure

```
Earthquake-Moment-Magnitude-Classifier/
├── .devcontainer/
│   └── devcontainer.json            # VS Code Dev Container configuration
├── app.py                           # Streamlit web application
├── classify.ipynb                   # Jupyter notebook for EDA and model training
├── earthquake_classifier_model.pkl # Trained machine learning model (PKL)
├── requirements.txt                 # Project dependencies
├── LICENSE                          # License file
└── README.md                        # Documentation
```

---

## 🛠️ Installation & Local Setup

### Prerequisites
- Python 3.8+ installed on your system
- Git installed

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Earthquake-Moment-Magnitude-Classifier.git
cd Earthquake-Moment-Magnitude-Classifier
```

### 2. Create and Activate a Virtual Environment

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### Launch the Streamlit App
To run the interactive web application locally:

```bash
streamlit run app.py
```

After executing the command, open your web browser and navigate to `http://localhost:8501`.

### Explore the Model Training Notebook
To inspect or retrain the machine learning model:

```bash
jupyter notebook classify.ipynb
```

---

## 🌐 Live Application

Access the hosted web app directly without setting up a local environment:
👉 **[Streamlit Live App](https://earthquake-moment-magnitude-classifier-mvkbcyyahegwfedqxkzdfv.streamlit.app/)**

---

## 🧰 Tech Stack

- **Programming Language:** Python
- **Web Framework:** [Streamlit](https://streamlit.io/)
- **Machine Learning:** Scikit-Learn
- **Data Handling & Visualization:** Pandas, NumPy, Matplotlib, Seaborn
- **Dev Tools:** Jupyter Notebooks, VS Code Dev Containers

---

## 📜 License

This project is open-source and available under the terms specified in the [LICENSE](LICENSE) file.
