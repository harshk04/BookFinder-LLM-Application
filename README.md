# BookFinder-LLM-Application

## Features

- **PDF Document Ingestion:** Load and split PDF documents into chunks for semantic search.
- **Embedding Generation:** Use HuggingFace's BAAI/bge-large-en model for generating embeddings.
- **Contextual Search:** Perform semantic searches to retrieve relevant document sections.
- **Response Generation:** Use GPT-2 from HuggingFace to generate responses based on the retrieved context.
- **Streamlit Interface:** User-friendly interface with navigation options for Home, Generate Response, and Contact Us pages.

## Requirements

- Python 3.8+
- Streamlit
- LangChain
- Qdrant
- HuggingFace Transformers
- Streamlit

## Installation

1. Clone the repository:

   `git clone https://github.com/your-repo/llama-gpt-contextual-search.git
   cd llama-gpt-contextual-search`

2. Install the required packages:
   
    `pip install -r requirements.txt`

3. Ensure Qdrant is running locally:
   
    `docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant`

## Usage
### Ingesting Data
1. Place your PDF document in the project directory and name it `Data.pdf`.
2. Run the `ingest.py` script to load the PDF, split it into chunks, and store the embeddings in Qdrant:

`python ingest.py`

### Running the Streamlit App
1. Run the Streamlit app:
   `streamlit run app.py`
2. Open your browser and go to `http://localhost:8501` to access the application.

## Project Structure

- **ingest.py**: Script to load PDF, split text, generate embeddings, and store them in Qdrant.
- **app.py**: Main application script with Streamlit interface for interacting with the search engine.
- **requirements.txt**: List of required Python packages.
- **Data.pdf**: Sample PDF document with the data of books.

## App Navigation

- **Home**: Overview of the application with an image and welcome message.
- **Generate Response**: Interactive chat interface to generate responses based on user input and document context.
- **Contact Us**: Form to contact the developer.

## License

This project is licensed under the `MIT License`.

## 📬 Contact

If you want to contact me, you can reach me through below handles.

&nbsp;&nbsp;<a href="https://www.linkedin.com/in/harsh-kumawat-069bb324b/"><img src="https://www.felberpr.com/wp-content/uploads/linkedin-logo.png" width="30"></img></a>

© 2024 Harsh
