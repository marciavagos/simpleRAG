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
from langchain_core.prompts import ChatPromptTemplate

class TransformersRAG:
  def __init__(self, model_name: str, use_faiss: bool = True):
    self.model_name _ model_name

  def build_index(self):
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

  def create_agent(self, query: str):
    system_context = "You are a stock market expert.\
    You will answer questions about Uber and Lyft companies as in the persona of a veteran stock market investor."

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_context,
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # Construct the Tools agent
    agent = create_tool_calling_agent(llm, tools, prompt,)
    
    # Create an agent executor by passing in the agent and tools
    self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True, handle_parsing_errors=True, max_iterations=10)

def query(self, query: str):
  response = self.agent_executor.invoke({"input": query})
  print("\nFinal Response:", response['output'])
  return response
