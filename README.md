# SLA-Guard Copilot
**Enterprise IT Ops Predictive Analytics & Azure OpenAI Decision Suite**

> **Live Demo:** [Launch Live App](https://sla-guard-copilot-u8hshu5yz6wrafptwdqvvx.streamlit.app/)

## Overview
SLA-Guard Copilot is an enterprise IT Operations decision suite that predicts Service Level Agreement (SLA) breaches before they occur and generates SOP-grounded resolution playbooks. It combines an **XGBoost predictive analytics pipeline** with a **RAG-powered SRE Copilot** built on the official Azure OpenAI-compatible SDK.

## Key Features
- **Ops Command Center:** Interactive Plotly KPIs tracking SLA compliance, category-wise breach rates, and Team vs. Priority risk heatmaps across 5,000 IT incidents.
- **Predictive SLA Breach Engine:** Scikit-learn + XGBoost pipeline (ColumnTransformer + OneHotEncoder + class-balanced XGBClassifier) predicting real-time breach probability with F1, Recall, and ROC-AUC metrics.
- **RAG Decision Copilot:** Lightweight TF-IDF cosine-similarity retriever over enterprise IT SOP runbooks (Database, Network/VPN, Cloud Infra, IAM, App Errors) paired with Pydantic-validated structured outputs.
- **Dual-Mode Azure OpenAI Architecture:** Supports enterprise **Azure OpenAI** and zero-cost **Azure-hosted GitHub Models (gpt-4o-mini)** via environment variables, with automatic deterministic SOP fallback.
- **FastAPI Microservice Layer:** Exposes `/health`, `/predict-sla`, and `/copilot/advise` REST endpoints.

## Tech Stack
- **ML & Analytics:** Python, Pandas, NumPy, Scikit-learn, XGBoost, Plotly
- **GenAI & RAG:** OpenAI / Azure OpenAI SDK, TF-IDF Vector Retrieval, Pydantic
- **Backend & UI:** FastAPI, Uvicorn, Streamlit

## Run Locally

    python -m venv venv
    .\venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    python src/generate_data.py
    python src/train_model.py
    streamlit run app/dashboard.py
