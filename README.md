# 🎭 SentimentScopes

> Real-time sentiment analysis on video subtitles using NLP — built with Streamlit and VADER.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-VADER-4CAF50?style=flat)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)

---

## 📌 Overview

**SentimentScopes** is a Streamlit-based web application that performs real-time sentiment analysis on video subtitles. It uses the **VADER (Valence Aware Dictionary and sEntiment Reasoner)** NLP model to classify the emotional tone of spoken content — giving content creators, researchers, and analysts instant insight into the emotional arc of any video.

---

## ✨ Features

- 📂 Upload subtitle files (`.srt`, `.txt`) and analyze instantly
- 🧠 Sentiment classification using VADER NLP (Positive / Negative / Neutral)
- 📊 Interactive visual dashboards with sentiment score breakdowns
- 🔄 Automated text preprocessing pipeline (cleaning, tokenization, normalization)
- 📈 Time-series sentiment trend view across subtitle segments

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend / UI | Streamlit |
| NLP Engine | VADER (via `vaderSentiment`) |
| Data Processing | Python, pandas |
| Visualization | Matplotlib / Streamlit charts |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/MahipathiRao/sentimentscopes.git
cd sentimentscopes

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Requirements

```
streamlit
vaderSentiment
pandas
matplotlib
```

---

## 📸 Screenshots

> _Add screenshots of the app UI here_

---

## 📂 Project Structure

```
sentimentscopes/
├── app.py                  # Main Streamlit application
├── utils/
│   ├── preprocessor.py     # Text cleaning & subtitle parsing
│   └── sentiment.py        # VADER sentiment analysis logic
├── sample_data/
│   └── sample.srt          # Sample subtitle file for testing
├── requirements.txt
└── README.md
```

---

## 🧠 How It Works

1. User uploads a subtitle file (`.srt` or `.txt`)
2. The preprocessing pipeline extracts and cleans raw text segments
3. VADER scores each segment on a scale from **-1 (negative)** to **+1 (positive)**
4. Results are visualized as sentiment classifications and trend graphs

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

**Venkata Mahipathi Rao Topella**  
📧 mahitthopella2004@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/mahi-topella) | [GitHub](https://github.com/MahipathiRao)
