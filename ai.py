import logging
import os
from dotenv import load_dotenv

# Import specific chat models from their respective libraries
from langchain_groq.chat_models import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI

# Import specific embeddings models from their respective libraries
from langchain_mistralai import MistralAIEmbeddings
from langchain_openai import OpenAIEmbeddings

#Import specific vector store database from their specific libraries
from pinecone import Pinecone, ServerlessSpec

load_dotenv()  # Load environment variables from .env file

from typing import List

class CompletionRequest:
    def __init__(self, dino_graphql: str, auth_token: str, namespace: str, info: List[str], chat: List[str]):
        self.dino_graphql = dino_graphql
        self.auth_token = auth_token
        self.namespace = namespace
        self.info = info
        self.chat = chat

class CompletionResponse:
    def __init__(self, error: str = None, paragraphs: List[str] = None, similarities: List[float] = None, answer: str = None):
        self.error = error
        self.paragraphs = paragraphs
        self.similarities = similarities
        self.answer = answer

def complete_chat(req: CompletionRequest, llm_type=None, model=None):                                                                                                                                        
     emb_llm_type = "OpenAI"
     llm_type = "OpenAI" 
     model = "gpt-4o-mini"
     emb_model ="text-embedding-ada-002"
     logging.info(f"Starting chat completion with llm_type: {llm_type}, model: {model}")                                                                                                                      
     if len(req.chat) % 2 == 0:                                                                                                                                                                               
         logging.error("Chat completion error: chat must be a list of user,assistant messages ending with a user message")                                                                                    
         return CompletionResponse(error="Chat completion error: chat must be a list of user,assistant messages ending with a user message")                                                                  
     question = req.chat[-1]                                                                                                                                                                                  
     logging.info(f"Processing question: {question}")                                                                                                                                                         
     logging.info(f"Namespace: {req.namespace}")                                                                                                                                                              
     paragraphs, similarities, err = find_similar_paragraphs(question, 2, 0.7, req.namespace, emb_llm_type=emb_llm_type, model=emb_model)                                                                             
     if err:                                                                                                                                                                                                  
         logging.error(f"Error finding similar paragraphs: {err}")                                                                                                                                            
         return CompletionResponse(error=f"Error finding similar paragraphs: {err}")                                                                                                                          
     if not req.info and not paragraphs:                                                                                                                                                                      
         logging.info("No information available for the question")                                                                                                                                            
         return CompletionResponse(answer="Non ho informazioni al riguardo")                                                                                                                                  
     logging.info(f"Found {len(paragraphs)} relevant paragraphs")                                                                                                                                             
                                                                                                                                                                                                              
     messages = [                                                                                                                                                                                             
         {"role": "system", "content": "You are Dino, an assistant who helps users by answering questions concisely."},                                                                          
         {"role": "user", "content": "In this chat, I will send you various information, followed by questions. Answer the questions with the information contained in this chat. If the answer is not contained in the information, answer 'I have no information about this'."},                                                                                                                                 
         {"role": "assistant", "content": "Ok!"},                                                                                                                                                             
     ]                                                                                                                                                                                                        
     for info in req.info:                                                                                                                                                                                    
         messages.append({"role": "user", "content": "Information:\n" + info})                                                                                                                               
     for info in paragraphs:                                                                                                                                                                                  
         messages.append({"role": "user", "content": "Information:\n" + info})                                                                                                                               
     messages.append({"role": "assistant", "content": "Information received!"})                                                                                                                              
     for i, msg in enumerate(req.chat):                                                                                                                                                                       
         role = "user" if i % 2 == 0 else "assistant"                                                                                                                                                         
         messages.append({"role": role, "content": msg})                                                                                                                                                      
                                                                                                                                                                                                              
     # Initialize the language model based on the provided type                                                                                                                                               
     if llm_type == 'Groq': 
         model_kwargs = {'seed': 26}
         llm = ChatGroq(model_name=model, temperature=0, api_key=os.environ['GROQ_API_KEY'], model_kwargs=model_kwargs)
     elif llm_type == 'Deepseek': 
         llm = ChatOpenAI(model_name=model, temperature=0, seed=26, base_url='https://api.deepseek.com', api_key=os.environ['DEEPSEEK_API_KEY']) 
     elif llm_type == 'Mistral':                                                                                                                                                                              
         llm = ChatMistralAI(model_name=model, temperature=0, seed=26, api_key=os.environ['MISTRAL_API_KEY'])                                                                                                 
     elif llm_type == 'OpenAI':                                                                                                                                                                               
         llm = ChatOpenAI(model_name=model, temperature=0, seed=26, api_key=os.environ['OPENAI_API_KEY'])                                                                                                     
                                                                                                                                                                                                              
     try:                                                                                                                                                                                                     
         resp = llm.invoke(messages)                                                                                                                                                                          
         print(resp)                                                                                                                                                                                          
         return CompletionResponse(answer=resp.content)                                                                                                                                                       
     except Exception as e:                                                                                                                                                                                   
         logging.error(f"Error in chat completion: {str(e)}")                                                                                                                                                 
         return CompletionResponse(error=f"Error in chat completion: {str(e)}")

def embed(emb_llm_type, model, text):
    embeddings = None
    logging.info(f"Attempting to embed text with {emb_llm_type} model: {model}")
    logging.debug(f"Text to embed (first 50 chars): {text[:50]}...")

    # Initialize the language model based on the provided type
    if emb_llm_type == 'Mistral':
        mistralai_api_key = os.getenv("MISTRAL_API_KEY")
        if not mistralai_api_key:
            logging.error("MISTRAL_API_KEY environment variable is not set")
            raise ValueError("MISTRAL_API_KEY environment variable is not set")
        try:
            embeddings = MistralAIEmbeddings(
                model=model,
                api_key=mistralai_api_key,
            )
            logging.info("MistralAIEmbeddings initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing MistralAIEmbeddings: {str(e)}")
            raise ValueError(f"Error initializing MistralAIEmbeddings: {str(e)}")
    elif emb_llm_type == 'OpenAI':
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logging.error("OPENAI_API_KEY environment variable is not set")
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        try:
            embeddings = OpenAIEmbeddings(
                model=model,
                api_key=openai_api_key,
            )
            logging.info("OpenAIEmbeddings initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing OpenAIEmbeddings: {str(e)}")
            raise ValueError(f"Error initializing OpenAIEmbeddings: {str(e)}")
    else:
        logging.error(f"Unsupported llm_type: {emb_llm_type}")
        raise ValueError(f"Unsupported llm_type: {emb_llm_type}")

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        logging.info("HF_TOKEN set in environment")

    try:
        logging.info("Attempting to embed query")
        single_vector = embeddings.embed_query(text)
        logging.info(f"Successfully embedded text with {emb_llm_type}")
        logging.debug(f"Embedded vector (first 5 elements): {single_vector[:5]}")
        return single_vector
    except Exception as e:
        logging.error(f"Error embedding text with {emb_llm_type}: {str(e)}")
        logging.error(f"Error type: {type(e).__name__}")
        if hasattr(e, 'response'):
            logging.error(f"Response content: {e.response.content}")
        if hasattr(e, '__dict__'):
            logging.error(f"Error attributes: {e.__dict__}")
        raise ValueError(f"Error embedding text with {emb_llm_type}: {str(e)}")

def connect_to_pinecone (index_name: str):
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)
    return index

def find_similar_paragraphs(text: str, top_k: int, min_similarity: float, namespace: str, emb_llm_type: str, model: str) -> tuple:
    logging.info(f"Finding similar paragraphs for text: {text[:50]}...")
    try:
        index = connect_to_pinecone("langchain-test-index")
        logging.info("Connected to Pinecone index")
        vec = embed(emb_llm_type, model, text)
        logging.info("Text embedded successfully")
        resp = index.query(
            vector=vec, 
            top_k=top_k, 
            include_metadata=True, 
            namespace=namespace,
            min_similarity=min_similarity
        )
        logging.info(f"Pinecone query completed, found {len(resp.matches)} matches")
        
        paragraphs = []
        similarities = []
        for vec in resp.matches:
            if vec.score >= min_similarity:
                paragraphs.append(vec.metadata["text"])
                similarities.append(vec.score)
        
        logging.info(f"Filtered to {len(paragraphs)} paragraphs above minimum similarity")
        return paragraphs, similarities, None
    except Exception as e:
        logging.error(f"Error in find_similar_paragraphs: {str(e)}")
        return [], [], str(e)

def reply_to_prompt(prompt):

    llm_type = "Groq"                                                                                                                                                                        
    model = "llama-3.1-70b-versatile"
    messages = [
        {"role": "system", "content": "Sei un esperto di monitoraggio e valutazione che supporta le Organizzazioni non governative a scrivere il proprio bilancio sociale."},
        {"role": "user", "content": prompt}
    ]
    # Initialize the language model based on the provided type                                                                                                                                               
    if llm_type == 'Groq':                                                                                                                                                                                   
        model_kwargs = {'seed': 26}                                                                                                                                                                          
        llm = ChatGroq(model_name=model, temperature=0, api_key=os.environ['GROQ_API_KEY'], model_kwargs=model_kwargs)                                                                                       
    elif llm_type == 'Deepseek':                                                                                                                                                                             
        llm = ChatOpenAI(model_name=model, temperature=0, seed=26, base_url='https://api.deepseek.com', api_key=os.environ['DEEPSEEK_API_KEY'])                                                              
    elif llm_type == 'Mistral':
        llm = ChatMistralAI(model_name=model, temperature=0, seed=26, api_key=os.environ['MISTRAL_API_KEY'])                                                                                                 
    elif llm_type == 'OpenAI':                                                                                                                                                                               
        llm = ChatOpenAI(model_name=model, temperature=0, seed=26, api_key=os.environ['OPENAI_API_KEY'])                                                                                                     
                                                                                                                                                                                                              
    try:                                                                                                                                                                                                     
        resp = llm.invoke(messages)
        print (resp.content)
        return CompletionResponse(answer=resp.content)
    except Exception as e:                                                                                                                                                                                   
         logging.error(f"Error in chat completion: {str(e)}")                                                                                                                                                 
         return CompletionResponse(error=f"Error in chat completion: {str(e)}")
    #try:
    #    response = openai.ChatCompletion.create(
    #        model="gpt-3.5-turbo",  # Replace with your completion model
    #        messages=messages
    #    )
    #   return response.choices[0].message.content
    #except openai.OpenAIError as e:
    #    raise Exception(f"Error computing chat completion: {e}")

