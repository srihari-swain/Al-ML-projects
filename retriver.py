import os
import re

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()


class Retriver:
    def __init__(self, vector_store, sources):
        os.environ["GROQ_API_KEY"] = os.getenv('GROQ_API_KEY')
        self.llm = init_chat_model("llama3-8b-8192", model_provider="groq")
        self.vector_store = vector_store
        self.sources = sources


    def answer_question(self, question, k=4):
        """Answer a question using the ingested content."""
        if not self.vector_store:
            return "Please ingest some URLs first."
        
        # Create a custom prompt that instructs the model to use only the retrieved content
        prompt_template = """
        You are a helpful assistant that answers questions strictly based on the retrieved content.
        
        Retrieved content:
        {context}
        
        Question: {question}
        
        Important instructions:
        1. Answer only using information from the retrieved content.
        2. If the answer cannot be found in the retrieved content, say "I cannot answer this based on the provided web content."
        3. Do not use any external knowledge or make assumptions.
        4. Cite the source URLs in your answer.
        
        Answer:
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create the retrieval QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": k}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        # Get the answer
        result = qa_chain.invoke({"query": question})
        
        # Extract source URLs from the retrieved documents
        source_docs = result.get("source_documents", [])
        unique_sources = set()
        
        for doc in source_docs:
            # Extract the source ID from the content
            content = doc.page_content
            source_match = re.search(r"Source: ([^\n]+)", content)
            if source_match:
                source_id = source_match.group(1).strip()
                if source_id in self.sources:
                    unique_sources.add(self.sources[source_id])
        
        # Add sources to the answer if not already included
        answer = result["result"]
        if unique_sources and "Source" not in answer:
            sources_text = "\n\nSources:\n" + "\n".join(unique_sources)
            answer += sources_text
        
        return answer