# 🧠 StressAI 2.0: BERT-Based Multi-Level Stress Assessment from Reddit Textual Data

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?logo=pytorch)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-black?logo=flask)
![License](https://img.shields.io/badge/License-MIT-green)

An AI-powered web application for **multi-level stress assessment** using **BERT**, **Natural Language Processing**, and **Deep Learning**.

</div>

---

## 📌 Overview

StressAI 2.0 is a deep learning-based web application that analyzes user-provided text and predicts the corresponding **stress severity level**.

Unlike traditional binary stress detection systems that classify text as either *Stress* or *No Stress*, StressAI 2.0 performs **three-level stress classification**:

- 🟢 Low Stress
- 🟡 Mild Stress
- 🔴 High Stress

The project leverages **BERT (Bidirectional Encoder Representations from Transformers)** to understand contextual language patterns and generate reliable stress predictions with confidence scores.

---

## ✨ Features

- 🧠 Fine-tuned BERT model for stress assessment
- 🟢 Low / 🟡 Mild / 🔴 High stress prediction
- 📊 Confidence score generation
- 📈 Interactive dashboard with Chart.js
- ⚡ Real-time inference using Flask REST API
- 🎨 Modern responsive UI built with HTML, CSS, Tailwind CSS & JavaScript
- 🔍 Context-aware NLP instead of keyword matching
- 🏗 Modular architecture for future extensions

---

# 🖥 System Architecture

```
User
   │
   ▼
Frontend Dashboard
(HTML + CSS + JS + Tailwind)

   │ REST API
   ▼

Flask Backend

   │
   ▼

Text Preprocessing
• Cleaning
• Tokenization
• Padding
• Attention Masks

   │
   ▼

BERT Tokenizer

   │
   ▼

Fine-tuned BERT Model

   │
   ▼

Softmax Probability Layer

   │
   ▼

Stress Classification

🟢 Low Stress
🟡 Mild Stress
🔴 High Stress

   │
   ▼

Confidence Score

   │
   ▼

Interactive Dashboard
```

---

# 🚀 Technologies Used

| Technology | Purpose |
|------------|----------|
| Python | Programming Language |
| PyTorch | Deep Learning |
| HuggingFace Transformers | BERT Model |
| Flask | Backend API |
| HTML5 | Frontend |
| CSS3 | Styling |
| Tailwind CSS | Responsive UI |
| JavaScript | Client-side Logic |
| Chart.js | Data Visualization |
| Pandas | Data Processing |
| NumPy | Numerical Computing |
| Scikit-learn | Model Evaluation |
| Git & GitHub | Version Control |

---

# 📊 Dataset

The project uses the **Dreaddit Dataset**, a publicly available Reddit dataset containing stress-related textual posts.

Dataset preprocessing includes:

- Text Cleaning
- Normalization
- Tokenization
- Padding
- Truncation
- Attention Mask Generation

---

# ⚙ Model Workflow

```
Input Text
      │
      ▼
Text Preprocessing
      │
      ▼
BERT Tokenizer
      │
      ▼
Fine-tuned BERT
      │
      ▼
Softmax Layer
      │
      ▼
Prediction Probabilities
      │
      ▼
Stress Level
(Low / Mild / High)
      │
      ▼
Confidence Score
      │
      ▼
Dashboard Visualization
```

---

# 📈 Output

The system predicts:

- Stress Level
- Confidence Score
- Class Probabilities
- Color-coded Severity Indicator
- Interactive Pie Chart

Example:

```
Input:

"I have multiple deadlines this week and I'm feeling overwhelmed."

Prediction:

🟡 Mild Stress

Confidence:
91.82%
