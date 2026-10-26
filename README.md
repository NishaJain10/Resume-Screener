# AI Resume Screener and Ranker

An AI-powered resume screening and ranking application built with **Python, Streamlit, Azure OpenAI, LangChain, and ChromaDB**. The application analyzes PDF resumes against a given job description, extracts candidate profiles, performs semantic similarity search, and uses an LLM to generate a ranked list of candidates with explanations.

## 🚀 Features

- **Resume Parsing:** Extracts candidate information from PDF resumes, including name, years of experience, current role, and technical skills.
- **AI-Powered Profile Extraction:** Uses Azure OpenAI and LangChain structured output to generate structured candidate profiles.
- **Semantic Search:** Uses ChromaDB to retrieve resumes based on their semantic similarity to a job description.
- **Intelligent Candidate Ranking:** Uses Azure OpenAI to re-rank candidates based on their profiles, semantic similarity scores, and the job description.
- **Persistent Vector Database:** Stores resume text, candidate metadata, and embeddings in ChromaDB for reuse across sessions.
- **Duplicate Detection:** Uses MD5 content hashes to avoid reprocessing unchanged resumes.
- **Candidate Comparison:** Displays a sortable table containing candidate experience, current role, skills, source resume, and semantic match score.
- **Resume Downloads:** Allows users to select a candidate and download their corresponding PDF resume.
- **Vector Search Inspection:** Provides an expandable section to inspect semantic search results and associated metadata.
- **Session State Management:** Retains completed ranking results during Streamlit reruns, including table sorting and candidate selection.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Application development |
| Streamlit | Interactive web interface |
| Azure OpenAI | Candidate profile extraction and AI-powered ranking |
| LangChain | LLM integration and PDF document loading |
| ChromaDB | Persistent vector storage and semantic search |
| Pydantic | Structured candidate profile validation |
| PyPDFLoader | PDF resume text extraction |
| python-dotenv | Environment variable management |
| uv | Python dependency and project environment management |

## ⚙️ How It Works

1. **Input:** The user provides a local folder path containing PDF resumes and a job description.
2. **Resume Loading:** The application discovers PDF files in the specified folder and extracts their text using `PyPDFLoader`.
3. **Profile Extraction:** Azure OpenAI extracts structured candidate information, including name, experience, technical skills, and current role.
4. **Data Indexing:** Resume text and candidate metadata are stored in a persistent ChromaDB collection.
5. **Semantic Search:** ChromaDB compares the job description against indexed resumes using cosine distance.
6. **AI Ranking:** Azure OpenAI evaluates the retrieved candidate profiles and generates a final ranking with brief explanations.
7. **Results Visualization:** Streamlit displays the ranking report and a sortable candidate comparison table.
8. **Resume Download:** Users can select a candidate and download the associated PDF resume.

## 📋 Prerequisites

Before running the application, ensure you have:

- Python 3.14 or newer, as specified in the project configuration.
- An Azure OpenAI resource with a deployed chat model compatible with LangChain structured output.
- `uv` installed for dependency management.
- Network access to Azure OpenAI and the default ChromaDB embedding model on first use.
- PDF resumes containing selectable text.

## 📦 Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-project-directory>
```

Replace the placeholders with your GitHub repository URL and project directory.

### 2. Install Dependencies

Install the project dependencies:

```powershell
uv sync
```

Install the additional packages required by the application:

```powershell
uv add langchain-community langchain-core langchain-openai python-dotenv
```

The application also uses Streamlit, ChromaDB, Pydantic, and other dependencies declared in the project configuration.

## 🔐 Configuration

Create a `.env` file in the application directory and add your Azure OpenAI configuration:

```dotenv
AZURE_OPENAI_DEPLOYMENT=your-deployment-name
AZURE_OPENAI_API_VERSION=your-api-version
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
```

Replace the placeholder values with your Azure OpenAI resource details.

**Important:**
- Use the Azure OpenAI deployment name configured in your resource.
- Ensure the API version is supported by your deployment.
- Keep your API key private.
- Never upload your `.env` file or other credentials to GitHub.
- Environment variables can also be supplied through the operating system.

All four environment variables are required because the Azure OpenAI model is initialized when the application starts.

## ▶️ Run the Application

Save the application script as `Resume-Screener.py` in the project directory.

Start the Streamlit application:

```powershell
uv run streamlit run ResumeScreener.py
```

Open the local URL displayed in the terminal.

### Using the Application

1. Enter the absolute path to the folder containing PDF resumes.
2. Paste the job description into the application.
3. Click **Analyze and Rank Resumes**.
4. Review the AI-generated ranking and candidate comparison table.
5. Select a candidate row to view the corresponding resume download option.
6. Download the selected candidate's PDF.
7. Expand **Vector DB matches** to inspect the underlying semantic search results.

The resume folder must be accessible from the machine running Streamlit. The application discovers lowercase `.pdf` files directly inside the selected folder and does not scan subdirectories.

## 📊 Understanding the Results

### AI-Generated Ranking

Azure OpenAI evaluates the retrieved candidate profiles against the job description and generates a ranked report with a brief explanation for each candidate.

### Candidate Comparison Table

The table presents candidate details, including:

- Candidate name
- Years of professional experience
- Current role
- Technical skills
- Source resume filename
- Semantic match score

The table is initially ordered by semantic similarity. Its order may differ from the final AI-generated ranking.

### Semantic Match Score

The semantic match score is calculated using:

```python
max(0, min(1, 1 - cosine_distance))
```

The resulting value is displayed as a percentage.

**Note:** This score represents a transformed semantic distance, not a probability of suitability or a calibrated hiring score.

### Vector Database Matches

The expandable section displays the semantic search results as JSON, including candidate metadata and similarity scores. It does not display raw embedding vectors.

## 🗄️ Data Storage

The application stores indexed resume data in a persistent ChromaDB database:

- **Database directory:** `./chroma_db`
- **Collection name:** `resumes`

The database stores extracted resume text, embeddings, candidate profile information, and source filenames.

### Indexing Behavior

- Unchanged resumes are skipped using their MD5 content hashes.
- Modified resumes generate new records; older records are not automatically replaced.
- Renamed or moved resumes with identical content may retain their previously indexed filenames.
- Existing records remain in the database when the selected resume folder changes.
- Removing a PDF from the source folder does not automatically remove its indexed record.
- Ranking queries can include candidates indexed during previous runs, not just candidates from the currently selected folder.

ChromaDB uses its default embedding function because the script does not configure an Azure OpenAI embedding deployment. The embedding model may need to be downloaded during initial setup.
