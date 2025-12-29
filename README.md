
# AI-Stock-Market-Analyzer
AI-powered stock market analyzer
=======
# 📊 AI-Powered Stock Market Analyzer

An end-to-end **AI + Quantitative Stock Analysis System** that combines technical indicators, regression-based trend detection, GPT-powered chart pattern recognition, and strategy backtesting, delivered through an interactive Streamlit dashboard.

This project demonstrates **real-world applied machine learning, quantitative finance, and system design**.

---

## 🚀 Features

### Quantitative Analysis
- EMA (20 / 50 / 200) based trend detection
- Momentum analysis
- Linear regression slope for trend strength
- Confidence-based trend classification (Bullish / Bearish / Neutral)

### AI Chart Pattern Analysis
- GPT-based multimodal analysis of stock chart images
- Detects bullish, bearish, or neutral patterns
- Provides pattern type and confidence score

### Signal Fusion
- Combines quantitative indicators with AI vision output
- Penalizes conflicting signals
- Produces final trend decision with confidence

### Strategy Backtesting
- EMA + regression-based trading strategy
- Stop-loss based risk management
- Performance metrics:
  - Strategy return
  - Buy & hold return
  - Win rate
  - Maximum drawdown

### Interactive Dashboard
- Built using Streamlit
- Real-time stock symbol input
- Clean metric-based visualization

---

## 🧠 System Flow

```
Stock Symbol Input
        ↓
Quantitative Analysis
        ↓
AI Chart Pattern Analysis
        ↓
Signal Fusion
        ↓
Backtesting
        ↓
Streamlit Dashboard
```

---

## 🛠 Tech Stack

- Python
- Streamlit
- yFinance
- NumPy, Pandas
- Scikit-learn
- OpenAI Multimodal Models

---

## ⚙️ Setup & Run

### 1. Clone Repository
```bash
git clone https://github.com/your-username/AI-Stock-Market-Analyzer.git
cd AI-Stock-Market-Analyzer
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file:
```env
OPENAI_API_KEY=your_api_key_here
```

### 4. Run Application
```bash
streamlit run app.py
```

---

## 📈 Use Cases

- AI-assisted stock trend analysis
- Quantitative strategy evaluation
- Algorithmic trading research
- Portfolio project for ML / AI / Quant roles

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**.  
It does not constitute financial advice.

---

## 👨‍💻 Author

**Chaitanya Pushkarna**  
AI / ML Engineer | Quantitative Analysis | Applied LLM Systems

