# System
import os

# Application
from openai import OpenAI
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, LLMPredictor, ServiceContext
# from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores import FAISSVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index import StorageContext, load_index_from_storage
from llama_index import SQLDatabase, GPTSQLStructStoreIndex
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

class TransformersRAG:
  def __init__(self, model_name: str, use_faiss: bool = True):
    self.model_name _ model_name

  def build_index(self, path_to_documents: str):
    # Load documents from a directory
    documents = SimpleDirectoryReader(path_to_documents).load_data()  # Load documents from a folder
    
    # Set up embedding and LLM predictor
    #embedding_model = OpenAIEmbedding()
    embedding_model = SentenceTransformer(self.model_name)
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                 device_map="auto")  # Adjust for GPU usage if available
    # self.model = OpenAI(model=self.model_name)
    
    llm_predictor = LLMPredictor(llm=self.model)

    if self.use_faiss:
      # Create a FAISS vector store and build the index
      vector_store = FAISSVectorStore(embedding_model=embedding_model)
      self.storage_context = StorageContext.from_defaults(vector_store=vector_store)
      index = GPTVectorStoreIndex.from_documents(documents,
                                               service_context=self.service_context,
                                               storage_context=self.storage_context)
    else:
      self.service_context = ServiceContext.from_defaults(embed_model=embedding_model,
                                                          llm_predictor=llm_predictor)
      index = GPTVectorStoreIndex.from_documents(documents,
                                                 embed_model=embedding_model)

    # Save the index for future use
    index.storage_context.persist(persist_dir='./index')
      
    self.index = index
    
  def query(self, query: str):
    # Load the saved index
    storage_context = StorageContext.from_defaults(persist_dir='./faiss_index')
    index = load_index_from_storage(storage_context)
    
    # Perform a query
    response = index.query(query, mode="embedding")
    print("Response:", response)

    return response

  def query_rag(self, query: str):
    # Define a query engine for RAG
    retrieved_docs = index.query(query, mode="embedding")
    context = "\n".join([doc.text for doc in retrieved_docs])
    print(f"Retrieved Context:\n{context}\n")
    
    generator = pipeline("text-generation",
                         model=self.model,self.tokenizer=tokenizer,
                         max_length=512)

    # Create a prompt for the model
    prompt = f"Use the following context to answer the question:\n\n{context}\n\nQ: {query}\nA:"
    response = generator(prompt, max_length=200, num_return_sequences=1)

    return response

  def query_from_db(self, db_path: str, query: str):
    # Connect to a SQL database
    sql_database = SQLDatabase(f"sqlite:///{db_path}")
    
    # Build an index for structured data
    sql_index = GPTSQLStructStoreIndex.from_documents([],
                                                      sql_database=sql_database,
                                                      service_context=self.service_context)
    
    # Query the SQL index
    response = sql_index.query(query)
    print("SQL Response:", response)

    return response
