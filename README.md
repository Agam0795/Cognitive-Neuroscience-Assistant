# 🧠 Cognitive Neuroscience Assistant

An intelligent web-based chatbot specializing in cognitive neuroscience, built with Python Flask and information retrieval techniques.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🌟 Features

- **Expert Knowledge Base**: Comprehensive information on:
  - 🧪 Neurotransmitters (dopamine, serotonin, GABA, glutamate, etc.)
  - 🧠 Brain regions (hippocampus, prefrontal cortex, amygdala, basal ganglia)
  - 🏥 Neurological disorders (ADHD, Parkinson's, Alzheimer's, depression, anxiety)
  - 💊 Treatments (medications, therapy, brain stimulation)
  - 🔬 Research methods (fMRI, EEG, PET, TMS)
  - 🎯 Cognitive functions (memory, attention, executive functions)

- **Intelligent Retrieval**: TF-IDF vectorization with cosine similarity for accurate responses
- **Two Modes**: 
  - **Tutor Mode**: Detailed, educational responses
  - **Concise Mode**: Brief, to-the-point answers
- **Beautiful UI**: Modern, responsive web interface with gradient design
- **Real-time Chat**: Async JavaScript for smooth interactions

## 🚀 Live Demo

[Try it live on Vercel](https://your-app-url.vercel.app) *(Coming soon)*

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Agam0795/Cognitive-Neuroscience-Assistant.git
   cd Cognitive-Neuroscience-Assistant
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python web_app.py
   ```

4. **Open in browser**
   ```
   http://localhost:5000
   ```

## 🎮 Usage

### Web Interface
1. Type your question in the input box
2. Press Enter or click Send
3. Switch between Tutor/Concise modes using the dropdown
4. Click example tags for quick questions

### CLI Mode
```bash
python app.py
```

## 💡 Example Questions

- "How does dopamine affect motivation?"
- "What causes Parkinson's disease?"
- "Explain ADHD treatments"
- "How do SSRIs work for depression?"
- "What is neuroplasticity?"
- "Compare EEG and fMRI"
- "What does the hippocampus do?"
- "Explain the amygdala's role in fear"

## 🏗️ Architecture

### Backend (`app.py`)
- **Knowledge Base**: 19 comprehensive documents covering neuroscience topics
- **Retrieval Engine**: TF-IDF vectorization + cosine similarity
- **Intent Detection**: Regex-based pattern matching
- **Response Generation**: Context-aware answer composition

### Web App (`web_app.py`)
- **Flask Server**: Handles HTTP requests
- **API Endpoints**: `/chat`, `/mode`
- **JSON Responses**: Structured data for frontend

### Frontend
- **HTML/CSS**: Modern, responsive design
- **JavaScript**: Vanilla JS with Fetch API
- **Features**: Real-time chat, markdown formatting, loading states

## 📦 Project Structure

```
cognitive-neuroscience-assistant/
├── app.py                 # Core backend logic
├── web_app.py            # Flask server
├── requirements.txt      # Python dependencies
├── templates/
│   └── index.html        # Main HTML page
├── static/
│   ├── style.css         # Styling
│   └── script.js         # Frontend logic
└── README.md             # This file
```

## 🔧 Technologies Used

- **Backend**: Python, Flask, scikit-learn, NumPy
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Algorithm**: TF-IDF, Cosine Similarity
- **Deployment**: Vercel (serverless)

## 📈 Future Enhancements

- [ ] Semantic search with sentence transformers
- [ ] Multi-turn conversation context
- [ ] Citation links to research papers
- [ ] User feedback system
- [ ] Export conversation history
- [ ] More imaging modalities
- [ ] Expanded disorder coverage

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Agam**
- GitHub: [@Agam0795](https://github.com/Agam0795)

## 🙏 Acknowledgments

- Built with information retrieval techniques for accuracy
- Knowledge base curated from neuroscience literature
- Inspired by the need for accessible neuroscience education

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

⭐ If you find this project useful, please consider giving it a star!
