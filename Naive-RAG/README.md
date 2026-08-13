## Plan: Build a GitHub Repository Code QA System with Hugging Face and FAISS

TL;DR: Build a Python app that ingests a GitHub repository, filters code and docs, chunks repository files, embeds chunks with a Hugging Face model, stores vectors in FAISS, and answers developer questions using a Hugging Face LLM with actual repo code as the source. Output should include the files and code chunks used to generate the answer.

## Quick start

```bash
pip install -r requirements.txt

# Ask a question against a GitHub repository
python src/app.py "owner/repo" --question "How does authentication work?"

# Ask against a local checkout
python src/app.py "data/repos/owner_repo" --question "What does predict() do?"

# Print the full prompt sent to the model
python src/app.py "owner/repo" --question "..." --verbose

# Use a different generation model
python src/app.py "owner/repo" --question "..." --model gpt2
```

The default generation model is `Qwen/Qwen3-4b` (instruction-tuned, good answers on CPU).
Small base models like `gpt2` run fast but cannot follow instructions, so answers will be poor.

### 1. Define the problem
- Input: GitHub repo URL or `owner/repo`
- Output: natural-language answers grounded in repository code
- Requirements:
  - use repository source files only
  - show file/chunk references with answers
  - support questions like:
    - “How does authentication work?”
    - “Where is the database connection initialized?”
    - “Explain the attendance deduplication algorithm.”
    - “Which files are responsible for camera log processing?”
    - “How does the frontend communicate with the backend?”

### 2. Repository ingestion
- Clone the GitHub repo locally or download the archive
- Support public repos initially
- Optionally support private repos with a GitHub token
- Keep a local working copy under `data/repos/<repo-name>`

### 3. File filtering
- Include code and doc file types:
  - `.py`, `.js`, `.ts`, `.tsx`, `.jsx`
  - `.java`, `.go`, `.rb`, `.php`
  - `.md`, `.rst`, `.txt`, `.yaml`, `.json`
- Exclude:
  - `.git`, `node_modules`, `venv`, `dist`, `build`, `__pycache__`, large binaries
- Optionally allow manual include/exclude patterns

### 4. Code/text chunking
- Chunk files into manageable pieces
- For code:
  - prefer function/class block chunks when possible
  - otherwise use sliding windows of 100–250 tokens
- For docs:
  - chunk by sections, headings, paragraphs
- Preserve metadata for each chunk:
  - repo path
  - start line / end line
  - language / file type
  - chunk id
  - original text

### 5. Embeddings
- Use a Hugging Face embedding model
- Recommended candidates:
  - `sentence-transformers/all-MiniLM-L6-v2`
  - code-specific models like `microsoft/codebert-base`
  - `hkunlp/instructor-base` if you want instruction-enhanced embeddings
- Compute embeddings for all chunks consistently
- Normalize vectors for cosine similarity

### 6. FAISS vector store
- Build a FAISS index for chunk embeddings
- Recommended index:
  - `IndexFlatIP` for cosine similarity with normalized vectors
  - or `IndexHNSWFlat` for larger repositories
- Store metadata in a parallel JSON or SQLite file keyed by chunk id
- Persist the index and metadata for reuse

### 7. Query flow
- Accept a developer question
- Embed the query using the same embedding model
- Search FAISS for top-K similar chunks
- Retrieve chunk metadata and text
- Rank or filter results if needed
- Build a prompt using the retrieved code context

### 8. Prompt construction
- Create a prompt template like:
  - “Use only the following repository code context.”
  - Include file path and line numbers for each chunk
  - Ask the model to answer only from the provided code
- Example structure:
  ```
  Context:
  [FILE: src/auth.py | LINES: 12-52]
  <code chunk>

  Question: How does authentication work in this project?
  Answer:
  ```

### 9. Hugging Face LLM
- Use a reasoning model appropriate for code QA
- Options:
  - local model with `transformers` + `accelerate`
  - Hugging Face Inference API if local GPU/CPU is limited
- Choose a model with enough capacity for code understanding

### 10. Output + attribution
- Return:
  - generated answer
  - list of source references:
    - file path
    - line range
    - relevance score
    - snippet preview
- Ensure the answer is grounded:
  - if code context is insufficient, the model should say “I don’t know” rather than hallucinate

### Project structure
- `src/repo_crawler.py` — clone/download GitHub repo
- `src/file_filter.py` — filter repo files by extension/path
- `src/chunker.py` — split files into chunks with line metadata
- `src/embeddings.py` — load HF embedding model and embed text
- `src/vector_store.py` — build/search FAISS index, persist metadata
- `src/qa.py` — retrieval, prompt creation, LLM query, answer assembly
- `src/app.py` or `src/server.py` — CLI or API entrypoint

### Verification
1. Confirm repo clone/download works for a sample GitHub repo
2. Confirm file filtering includes relevant code/docs
3. Confirm chunking preserves line ranges
4. Confirm FAISS search returns expected file chunks for sample queries
5. Confirm the model answer includes actual repo references

### Decisions
- Language: Python
- Vector store: FAISS
- Embeddings: Hugging Face embedding model
- Reasoning: Hugging Face LLM
- Source: actual repository code only

### Further improvements
- add a code-aware parser for functions/classes
- add reranking or hybrid search
- add a simple web UI
- support private GitHub repos
- support multi-repo projects and dependency awareness
