<div align="center">

# 🌍 ClimateEmotionLab-NLP

**Advanced NLP Pipeline for Multi-Modal Emotion Analysis in Climate Change Discourse**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-4.20+-yellow.svg)](https://huggingface.co/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*Analyzing emotional responses to climate change across news media and social platforms using state-of-the-art transformer models.*

[Overview](#-overview) •
[Features](#-features) •
[Installation](#-installation) •
[Usage](#-usage) •
[Results](#-results) •
[Citation](#-citation)

</div>

---

## 📖 Overview

**ClimateEmotionLab-NLP** is a comprehensive research project that leverages advanced Natural Language Processing techniques to understand how people emotionally respond to climate change across different media channels.

This project addresses a critical gap in climate communication research by moving beyond simple sentiment analysis (positive/negative) to fine-grained **27-category emotion classification** using the GoEmotions framework, enabling nuanced insights into public perception of climate issues.

### 🎯 Research Questions

1. **Cross-Media Comparison**: How do emotional expressions differ between professional news media and social media when discussing climate change?
2. **Emotion-Sentiment Mapping**: What is the relationship between sentiment polarity and fine-grained emotion categories?
3. **Engagement Predictors**: Which emotions are most predictive of social media engagement on climate topics?
4. **Temporal Dynamics**: How do temporal patterns in climate emotions vary across different media sources?

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔬 **Multi-Dataset Integration** | Combines RSS news headlines, pre-labeled sentiment data, and climate tweets (10,000+ texts) |
| 🧠 **Fine-Tuned RoBERTa** | Domain-adapted GoEmotions classifier for climate-specific text |
| 📊 **Cross-Media Analysis** | Statistical comparison of emotions across news vs. social media |
| 🔍 **Model Interpretability** | SHAP-based explainability with attention visualization |
| 📈 **Statistical Validation** | Chi-square, ANOVA, KL-divergence for emotion distribution analysis |
| 📉 **Interactive Dashboards** | Publication-ready visualizations with Plotly |

---

## 🏗️ Project Structure

```
ClimateEmotionLab-NLP/
├── 📁 climate_emotion_analysis/
│   ├── 📁 notebooks/
│   │   ├── 01_collect_headlines_rss.ipynb      # RSS data collection
│   │   ├── 02_combine_clean_headlines.ipynb    # Data preprocessing
│   │   ├── 03_clean_climate_tweets.ipynb       # Tweet cleaning pipeline
│   │   ├── 04_integrate_all_datasets.ipynb     # Multi-source integration
│   │   ├── 05_train_emotion_classifier.ipynb   # RoBERTa fine-tuning
│   │   ├── 05b_domain_adaptation.ipynb         # Climate domain adaptation
│   │   ├── 06_apply_emotion_analysis.ipynb     # Inference pipeline
│   │   ├── 07_sentiment_emotion_validation.ipynb # Validation analysis
│   │   ├── 08_cross_media_analysis.ipynb       # Media comparison
│   │   └── 09_interpretability_analysis.ipynb  # SHAP & explainability
│   ├── 📁 src/                                 # Source modules
│   ├── 📁 paper/                               # Research paper drafts
│   └── requirements.txt
├── 📁 Datasets/                                # Raw data (gitignored)
└── README.md
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended for training)
- 16GB+ RAM

### Setup

```bash
# Clone the repository
git clone https://github.com/Zuraiz270/ClimateEmotionLab-NLP.git
cd ClimateEmotionLab-NLP

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r climate_emotion_analysis/requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

---

## 🚀 Usage

### 1️⃣ Data Preparation

Run notebooks `01` through `04` sequentially to collect and preprocess data:

```bash
cd climate_emotion_analysis/notebooks
jupyter notebook 01_collect_headlines_rss.ipynb
```

### 2️⃣ Model Training

Fine-tune the RoBERTa emotion classifier:

```bash
jupyter notebook 05_train_emotion_classifier.ipynb
```

### 3️⃣ Emotion Analysis

Apply the trained model to all datasets:

```bash
jupyter notebook 06_apply_emotion_analysis.ipynb
```

### 4️⃣ Cross-Media Analysis

Generate statistical comparisons and visualizations:

```bash
jupyter notebook 08_cross_media_analysis.ipynb
```

---

## 📊 Results

### Emotion Distribution Across Media Types

| Emotion Category | News Headlines | Social Media |
|-----------------|----------------|--------------|
| Fear / Anxiety | High | Moderate |
| Anger | Moderate | High |
| Sadness | Moderate | Moderate |
| Hope / Optimism | Low | Moderate |

> *Detailed results and visualizations available in `notebooks/08_cross_media_analysis.ipynb`*

### Model Performance

- **Base GoEmotions F1-Score**: ~0.46 (27-class multi-label)
- **Domain-Adapted F1-Score**: Improved performance on climate-specific text

---

## 🛠️ Tech Stack

<div align="center">

| Category | Technologies |
|----------|-------------|
| **Deep Learning** | PyTorch, Transformers (HuggingFace) |
| **NLP** | spaCy, NLTK, RoBERTa |
| **Data Science** | Pandas, NumPy, Scikit-learn |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Explainability** | SHAP |
| **Statistics** | SciPy, Statsmodels |

</div>

---

## 📚 Datasets

This project integrates multiple data sources:

1. **RSS Climate Headlines**: Real-time news collection from major outlets
2. **Sentiment-Labeled Headlines**: Pre-annotated climate news dataset
3. **Climate Change Tweets**: Social media discourse (8,900+ tweets)
4. **GoEmotions**: Google's 58k Reddit comments for emotion classifier training

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@misc{climateemotionlab2024,
  author = {Zuraiz},
  title = {ClimateEmotionLab-NLP: Multi-Modal Emotion Analysis in Climate Change Discourse},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/Zuraiz270/ClimateEmotionLab-NLP}
}
```

---

<div align="center">

**Built with ❤️ for Climate Research**

*Advanced NLP Practicum Project*

</div>
