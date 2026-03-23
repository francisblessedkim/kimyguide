```markdown
# KimyGuide 🎓

KimyGuide is a goal-based educational recommender system designed to generate meaningful course recommendations under **cold-start conditions**, where no prior user data is available.

The system combines:
- **TF-IDF (lexical model)** → traditional keyword-based approach  
- **Sentence-BERT embeddings (semantic model)** → semantic understanding  
- **Hybrid model** → combines both for improved ranking performance  

---

## 🚀 Features

- Goal-based recommendations (e.g. "I want to learn Spanish")
- Model comparison interface (TF-IDF vs Embeddings vs Hybrid)
- FastAPI backend with interactive UI
- Cold-start focused (no user history required)

---

## 📂 Project Structure

```

.
├── src/
│   └── kimyguide/
│       ├── api/              # FastAPI app
│       ├── models/           # TF-IDF, embeddings, hybrid logic
│       ├── features/         # Text processing
│       ├── explainers/       # Explanation logic
│
├── data/
│   └── raw/                  # OpenLearn dataset
│
├── scripts/                  # Dataset scripts (scraping / prep)
├── run_cli.py
├── requirements.txt
└── README.md

````

---

## ⚙️ Prerequisites

- Python 3.9+
- pip

---

## 🧰 Setup (macOS / Linux)

```bash
git clone https://github.com/your-username/kimyguide.git
cd kimyguide

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
````

---

## 🧰 Setup (Windows - PowerShell)

```powershell
git clone https://github.com/your-username/kimyguide.git
cd kimyguide

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## 📊 Dataset Setup

Place your dataset here:

```
data/raw/openlearn_courses.csv
```

If you have a scraping script:

```bash
python scripts/scrape_openlearn.py
```

---

## ▶️ Run the Application

⚠️ IMPORTANT: Your project uses a `src/` structure, so you must include `PYTHONPATH=src`

```bash
PYTHONPATH=src uvicorn kimyguide.api.app:app --reload --host 127.0.0.1 --port 8000
```

---

## 🌐 Access the App

* Main UI: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
* Compare Models: [http://127.0.0.1:8000/ui/compare](http://127.0.0.1:8000/ui/compare)
* API Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 💡 Example Usage

Input:

```
I want to learn Spanish
```

The system returns:

* TF-IDF results (keyword-based)
* Embedding results (semantic)
* Hybrid results (combined)

---

## 🧪 Run CLI Demo

```bash
python run_cli.py --data-path data/raw/openlearn_courses.csv
```

---

## 🧪 Running Tests

```bash
export KIMYGUIDE_SKIP_EMBEDDINGS=1
pytest -q
```

---

## ⚠️ Troubleshooting

### Imports not working?

Make sure you are using:

```bash
PYTHONPATH=src
```

---

### Embeddings too slow?

Disable them:

```bash
export KIMYGUIDE_SKIP_EMBEDDINGS=1
```

---

## 📌 Notes

* Designed for **cold-start recommendation**
* No user history required
* Uses only user request + course data

---

## 📄 License

MIT License
