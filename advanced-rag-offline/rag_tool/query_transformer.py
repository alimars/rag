from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import json
import os

class QueryTransformer:
    def __init__(self):
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        query_transformer_model = os.getenv("QUERY_TRANSFORMER_MODEL", "llama3:8b")
        self.llm = OllamaLLM(model=query_transformer_model, base_url=ollama_base_url, temperature=0.3)
    
    def multi_query(self, query):
        prompt = ChatPromptTemplate.from_template("""
        Generate 3 different versions of the user's question for document retrieval.
        Focus on different aspects and synonyms. Return ONLY a JSON array:
        ["query1", "query2", "query3"]
        
        Original: {query}""")
        chain = prompt | self.llm
        try:
            return json.loads(chain.invoke({"query": query}).strip())
        except:
            return [query]
    
    def decompose_query(self, query):
        prompt = ChatPromptTemplate.from_template("""
        Break this complex question into 2-4 standalone sub-questions. 
        Return ONLY a JSON array: ["subq1", "subq2", ...]
        
        Question: {query}""")
        chain = prompt | self.llm
        try:
            return json.loads(chain.invoke({"query": query}).strip())
        except:
            return [query]