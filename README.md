# 🧠 Cognitive Neuroscience Assistant

An intelligent chatbot powered by information retrieval that provides expert knowledge about cognitive neuroscience, brain function, neurological disorders, and treatments.

## 🌐 Live Demo

**Deployed on Vercel:** [https://cognitive-neuroscience-assistant-hv10k5b4t.vercel.app](https://cognitive-neuroscience-assistant-hv10k5b4t.vercel.app)

## ✨ Features

### Comprehensive Knowledge Base
- **Neurotransmitters**: Dopamine, serotonin, GABA, glutamate, acetylcholine, norepinephrine, endorphins, oxytocin
- **Brain Regions**: Prefrontal cortex, hippocampus, amygdala, basal ganglia, and their functions
- **Disorders**: ADHD, Parkinson's, Alzheimer's, depression, anxiety, schizophrenia, PTSD
- **Treatments**: Medications (SSRIs, stimulants, antipsychotics), therapy (CBT, DBT), brain stimulation (TMS, ECT)
- **Research Methods**: fMRI, EEG, MEG, PET, TMS, experimental design
- **Cognitive Functions**: Memory systems, attention, executive functions, decision-making

### Intelligent Features
- **TF-IDF Retrieval**: Fast, accurate information retrieval using scikit-learn
- **Intent Recognition**: Detects query types (definitions, comparisons, treatments, etc.)
- **Dual Response Modes**: Toggle between detailed "Tutor" and brief "Concise" answers
- **60+ Glossary Terms**: Quick definitions for neuroscience terminology
- **13 FAQ Templates**: Pre-answered common questions

### Modern Web Interface
- Beautiful gradient UI with smooth animations
- Real-time chat interface
- Example question tags for quick exploration
- Responsive design (mobile & desktop)
- Markdown formatting support

## 🏗️ Architecture

### Backend (`app.py`)
- **Knowledge Base**: 19 comprehensive documents covering neuroscience topics
- **Retrieval Engine**: TF-IDF vectorization with cosine similarity scoring
- **Intent Detection**: Regex-based pattern matching for query classification
- **Response Generation**: Context-aware answer composition with intent-specific tips

### Web Application (`web_app.py`, `index.py`)
- **Flask Server**: RESTful API with `/chat` and `/mode` endpoints
- **Serverless**: Optimized for Vercel deployment

### Frontend
- **HTML/CSS/JavaScript**: Clean, vanilla implementation
- **No external dependencies**: Fast loading, no bloat
- **Async API calls**: Non-blocking user experience

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/Agam0795/Cognitive-Neuroscience-Assistant.git
cd Cognitive-Neuroscience-Assistant
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the web application**
```bash
python web_app.py
```

4. **Open in browser**
```
http://localhost:5000
```

### CLI Mode

For command-line usage:
```bash
python app.py
```

## 📦 Dependencies

- **Flask 2.3.2**: Web framework
- **NumPy**: Numerical computing
- **scikit-learn**: TF-IDF vectorization and similarity calculations

## 🎯 Example Questions

- "How does dopamine affect motivation and reward?"
- "What causes Parkinson's disease and how is it treated?"
- "Explain ADHD neurobiological basis and treatments"
- "How do SSRIs work for depression?"
- "What is neuroplasticity and how can I enhance it?"
- "Compare EEG and fMRI for research"
- "What's the role of the amygdala in fear and anxiety?"
- "Explain the hippocampus and memory formation"

## 🧪 How It Works

### 1. **TF-IDF Vectorization**
```python
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
```
Converts documents and queries into numerical vectors based on term frequency and document rarity.

### 2. **Cosine Similarity**
```python
similarity = cosine_similarity(query_vector, document_vectors)
```
Calculates the angle between vectors to find the most relevant passages.

### 3. **Response Composition**
- Retrieves top 3 relevant passages
- Checks FAQ database for direct matches
- Adds intent-specific tips
- Formats based on user's mode preference

## 🌍 Deployment

### Vercel (Current)

Automatically deployed from the `main` branch:
```bash
vercel --prod
```

### Manual Deployment Options

**Docker:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "web_app.py"]
```

**Heroku:**
```bash
heroku create cognitive-neuro-assistant
git push heroku main
```

## 📊 Project Statistics

- **19 Knowledge Documents**: Comprehensive expert-level content
- **60+ Glossary Terms**: Quick reference definitions
- **13 FAQ Entries**: Common questions pre-answered
- **10 Intent Patterns**: Smart query classification
- **~500 lines** of core logic in `app.py`

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Expand Knowledge Base**: Add more documents to `KB_DOCS`
2. **Add Glossary Terms**: Enhance `GLOSSARY` dictionary
3. **Improve Intent Detection**: Add patterns to `INTENT_PATTERNS`
4. **UI Enhancements**: Improve frontend design
5. **Bug Fixes**: Report and fix issues

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Agam Singh**
- GitHub: [@Agam0795](https://github.com/Agam0795)
- Project: [Cognitive-Neuroscience-Assistant](https://github.com/Agam0795/Cognitive-Neuroscience-Assistant)

## 🙏 Acknowledgments

- Information retrieval techniques from scikit-learn
- Flask web framework
- Vercel for serverless deployment
- Cognitive neuroscience research community

## 🔮 Future Enhancements

- [ ] Semantic search using sentence transformers
- [ ] Multi-turn conversation context
- [ ] User feedback system
- [ ] Citation links to research papers
- [ ] Voice input/output
- [ ] Multi-language support
- [ ] Mobile app version

---

**Made with 🧠 and ❤️ for the neuroscience community**

*This is an educational tool. Always consult healthcare professionals for medical advice.*
