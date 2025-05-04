from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import init_chat_model

from scrapper import  scrape_url

class DB_urlingection:
    def __init__(self):
        self.sources = {}
        self.vector_store = None
        self.embeddings = HuggingFaceBgeEmbeddings(model_name='all-MiniLM-L6-v2')
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )

    def ingest_urls(self, urls):
        """Ingest content from multiple URLs into the vector store."""
        all_texts = []
        url_mapping = {}  # To keep track of which chunk came from which URL
        
        for url in urls:
            content = scrape_url(url)
            if content.startswith("Error scraping"):
                print(content)
                continue
            
            # Split the content into chunks
            chunks = self.text_splitter.split_text(content)
            
            # Store the source URL for each chunk
            for i, chunk in enumerate(chunks):
                chunk_id = f"{url}_{i}"
                url_mapping[chunk_id] = url
                # Add a source prefix to help with retrieval
                all_texts.append(f"Source: {chunk_id}\n\n{chunk}")
                self.sources[chunk_id] = url
        
        if not all_texts:
            return "No content was successfully ingested."
        
        # Create or update the vector store
        if self.vector_store is None:
            self.vector_store = FAISS.from_texts(all_texts, self.embeddings)
        else:
            temp_db = FAISS.from_texts(all_texts, self.embeddings)
            self.vector_store.merge_from(temp_db)
        
        return self.vector_store, self.sources