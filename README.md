# 🏁 F1 Race Intelligence Copilot

An interactive **ML + LLM** web app that predicts Formula 1 lap times, simulates pit strategy timing, and compares constructors — explained in natural language like an F1 race engineer.

---

## 🚀 Features
- Predict lap times from 600k+ historical race laps  
- Simulate pit timing changes (early / late pit)  
- Compare two constructors (Team A vs Team B)  
- GPT-powered race engineer explanations (Markdown)  
- Interactive Gradio UI  

---

## 🧠 Tech Stack
Python · Pandas · Scikit-learn · OpenAI API · Gradio

---

## 📊 Data
Public Formula 1 datasets (races, results, lap times, pit stops, constructors, circuits).

---
## ▶️ Run Locally

```bash
git clone https://github.com/Noel2521/F1-Race-Intelligence-Copilot.git
cd F1-Race-Intelligence-Copilot
pip install -r requirements.txt
python app.py


Also add this small note:

```markdown
## 🔐 Environment Variables

Create a `.env` file and add:

```bash
OPENAI_API_KEY=your_api_key_here