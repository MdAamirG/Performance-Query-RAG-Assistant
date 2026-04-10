# Quick Start Guide

## 🚀 5-Minute Setup

### 1. Get Gemini API Key (Free)
```bash
# Visit: https://makersuite.google.com/app/apikey
# Create new API key and copy it
```

### 2. Setup Project
```bash
cd /Users/mohamedaamir/Desktop/solartis

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Key
```bash
# Copy the example and add your key
cp .env.example .env

# Edit .env with your Gemini API key
# GEMINI_API_KEY=your_api_key_here
```

### 4. Test the System
```bash
# Run test suite (no API key needed for this)
python tests/test_rag_pipeline.py

# Run interactive analyzer (requires API key in .env)
cd src
python main.py

# Run the Streamlit demo UI
streamlit run app.py
```

## 🎯 Try These Queries

Once in interactive mode, try asking:

```
"Why is a full table scan slow?"
"How to optimize JSON field filtering?"
"What causes database latency spikes?"
"How to handle high-frequency queries?"
"Why does this complex join query slow down?"
```

Type `list` to see all available cases.
Type `exit` to quit.

## 📊 Project Structure

```
solartis/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration & prompts
│   ├── rag_pipeline.py        # Vector DB & retrieval
│   ├── ai_analyzer.py         # Gemini integration
│   └── main.py                # Main orchestrator & CLI
├── data/
│   ├── performance_cases.json # 10 real-world scenarios
│   └── chroma_db/             # Vector DB (auto-created)
├── tests/
│   └── test_rag_pipeline.py   # Comprehensive tests
├── README.md                  # Full documentation
├── requirements.txt           # Python dependencies
├── .env.example               # API key template
└── .gitignore                 # Git configuration
```

## 🔧 Running Different Ways

### Interactive Mode (Recommended for Testing)
```bash
cd src
python main.py
```

### Streamlit Demo UI
```bash
streamlit run app.py
```

### Batch Mode (Multiple Queries)
```bash
cd src
python main.py \
  "Why is SELECT * FROM large_table slow?" \
  "How to optimize JSON extraction?"
```

### Programmatic Usage
```python
from main import PerformanceAnalysisSystem

system = PerformanceAnalysisSystem()
result = system.analyze_query("Why is my query slow?")
print(system.format_response(result))
```

## ⚡ Performance

- Embedding: ~5-10ms per query
- Vector retrieval: ~2-3ms
- Gemini API: ~1-2 seconds
- **Total:** ~2-3 seconds end-to-end

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'google.genai'"
```bash
source venv/bin/activate
pip install google-genai
```

### "GEMINI_API_KEY is not set"
1. Create `.env` file in project root
2. Add: `GEMINI_API_KEY=your_key_here`
3. Get free key: https://makersuite.google.com/app/apikey

### "Chroma DB error"
```bash
rm -rf data/chroma_db
python main.py  # Will recreate
```

### Tests fail
```bash
source venv/bin/activate
python tests/test_rag_pipeline.py
```

## 📚 Next Steps

1. **Setup:** Follow 5-minute setup above
2. **Test:** Run `python tests/test_rag_pipeline.py`
3. **Explore:** Try interactive mode with sample queries
4. **Integrate:** Add your Gemini API key for full AI analysis
5. **Customize:** Modify prompts in `src/config.py` for your use case

---

See [README.md](README.md) for complete architecture and design documentation.
