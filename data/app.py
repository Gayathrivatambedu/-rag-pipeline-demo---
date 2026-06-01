
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import os

from dotenv import load_dotenv
load_dotenv()

# 1. Load documents
loader = TextLoader("data/sample.txt")
documents = loader.load()

# 2. Chunk text
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# 3. Create embeddings + vector store
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(chunks, embeddings)

# 4. Create retriever
retriever = vectordb.as_retriever()

# 5. Build RAG chain
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    retriever=retriever,
    return_source_documents=True
)

# 6. Ask a question
query = "What is this document about?"
result = qa(query)

print("\nAnswer:", result["result"])
print("\nSources:", result["source_documents"])
