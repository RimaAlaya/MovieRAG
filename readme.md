# 🎬 CineRAG

**CineRAG** is a Retrieval-Augmented Generation (RAG) system for querying movie and TV series information using semantic search and local LLM inference.

## ✨ Features

- **Semantic Search**: Uses sentence transformers to understand natural language queries
- **Vector Database**: FAISS for efficient similarity search across movie/TV content
- **Local LLM Integration**: Mistral 7B Instruct for generating contextual answers
- **Streamlit UI**: Simple web interface for interactive queries
- **Extensible Knowledge Base**: JSON-based content storage for easy updates

## 📋 Current Knowledge Base

- Friends (TV Series)
- How I Met Your Mother (TV Series)
- Inception (Movie)
- Stranger Things (TV Series)

Each entry includes:
- Character descriptions
- Plot summaries
- Trivia & fun facts
- Themes
- Ratings and awards

## 🚀 Installation

### Prerequisites

- Python 3.12+
- CUDA-compatible GPU (recommended) or CPU
- Hugging Face account (for model access)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/cinerag.git
cd cinerag
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up Hugging Face authentication**
```bash
huggingface-cli login
```
Enter your Hugging Face token when prompted.

5. **Run the application**
```bash
streamlit run app.py
```

## 📦 Dependencies

Create a `requirements.txt` file with:

```
streamlit==1.29.0
sentence-transformers==2.2.2
faiss-cpu==1.7.4  # or faiss-gpu for GPU support
transformers==4.36.0
torch==2.1.0
numpy==1.24.3
accelerate==0.25.0
```

## 🎯 Usage

### Web Interface

1. Start the Streamlit app: `streamlit run app.py`
2. Enter your question in the text input
3. Receive AI-generated answers based on retrieved context

### Python API

```python
from main import rag_answer, query_system

# Get raw retrieval results
results = query_system("Who plays Eleven in Stranger Things?", top_k=3)

# Get LLM-generated answer
answer = rag_answer("What is Inception about?", top_k=2)
print(answer)
```

## 📁 Project Structure

```
cinerag/
├── app.py                  # Streamlit web interface
├── main.py                 # Core RAG logic and LLM integration
├── docs/                   # Knowledge base (JSON files)
│   ├── Friends.json
│   ├── HowIMetYourMother.json
│   ├── Inception.json
│   └── StrangerThings.json
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🔧 How It Works

1. **Document Loading**: JSON files are parsed and chunked by content type
2. **Embedding Creation**: Text chunks are converted to vectors using `all-MiniLM-L6-v2`
3. **Vector Storage**: FAISS indexes embeddings for fast similarity search
4. **Query Processing**: User questions are embedded and matched against the index
5. **Context Retrieval**: Top-k most relevant chunks are retrieved
6. **Answer Generation**: Mistral 7B generates answers using retrieved context

## 🎨 Adding New Content

Create a JSON file in the `docs/` folder:

```json
{
  "title": "Your Movie/Show Title",
  "chunks": [
    {
      "type": "trivia",
      "text": "Interesting facts go here..."
    },
    {
      "type": "characters",
      "text": "Character descriptions..."
    },
    {
      "type": "plot",
      "text": "Plot summary..."
    },
    {
      "type": "themes",
      "text": "Major themes..."
    },
    {
      "type": "ratings",
      "text": "IMDb, Rotten Tomatoes, etc."
    }
  ]
}
```

Restart the application to load new content.

## ⚙️ Configuration

### Model Selection

To use a different LLM, modify `main.py`:

```python
model_name = "your-model-name"  # e.g., "meta-llama/Llama-2-7b-chat-hf"
```

### Retrieval Settings

Adjust `top_k` parameter for more/fewer retrieved chunks:

```python
answer = rag_answer(question, top_k=5)  # Default: 3
```

### GPU/CPU Toggle

For CPU-only inference (slower):
```python
llama_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  # Change to float32
    # Remove device_map="auto"
)
```

## 🐛 Troubleshooting

**Out of Memory Error**
- Reduce `top_k` value
- Use a smaller model
- Enable CPU offloading with `device_map="auto"`

**Slow Generation**
- Use GPU if available
- Reduce `max_new_tokens` in generation config
- Consider quantized models (4-bit, 8-bit)

**Model Access Issues**
- Ensure you're logged into Hugging Face: `huggingface-cli login`
- Accept model terms on Hugging Face website if required

## 🚀 Future Enhancements

- [ ] Add more movies and TV series
- [ ] Implement conversation memory
- [ ] Support for images and posters
- [ ] Advanced filtering (genre, year, rating)
- [ ] Export/save favorite responses
- [ ] Multi-language support
- [ ] RESTful API endpoint

## 📄 License

MIT License - feel free to use and modify for your projects.

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add your content or improvements
4. Submit a pull request

## 📧 Contact

For questions or suggestions, open an issue on GitHub.

---

**Built with** 🧠 Transformers • 🔍 FAISS • 🎨 Streamlit