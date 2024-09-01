# Import necessary libraries for the Flask application
import os
import logging
from flask import Flask, request, jsonify, abort
from flask_cors import CORS
import pandas as pd
from pandasai import Agent
import database
from database import validate_api_key

import dino
from dino import dino_authenticate

import ai 
from ai import complete_chat, CompletionResponse

# Import specific chat models from their respective libraries
from langchain_groq.chat_models import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI


# Initialize the Flask application
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

@app.after_request
def add_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    return response

# Define a route for the '/' endpoint that returns a welcome message
@app.route('/')
def welcome():
    return "Welcome to Pandino! This is the root endpoint."

def validate_api_key(api_key):
    if not api_key:
        abort(403)
    result, message = database.validate_api_key(api_key)
    if not result:
        if "expired" in message:
            abort(403, description="API key expired")
        else:
            abort(403, description="Invalid API key")

@app.route("/completion.json", methods=["POST"])
def completion_handler():
    try:
        r = request.get_json()
        if not r:
            return jsonify({"error": "No JSON data provided"}), 400

        required_keys = ["dinoGraphql", "authToken", "chat"]
        missing_keys = [key for key in required_keys if key not in r]

        if missing_keys:
            return jsonify({"error": f"Missing required keys: {', '.join(missing_keys)}"}), 400

        err = dino_authenticate(r["dinoGraphql"], r["authToken"])
        if err:
            return jsonify({"error": f"Authentication error: {str(err)}"}), 401

        # Prepare the request for complete_chat
        chat_request = ai.CompletionRequest(
            dino_graphql=r["dinoGraphql"],
            auth_token=r["authToken"],
            namespace=r.get("namespace", ""),
            info=r.get("info", []),
            chat=r["chat"]
        )

        resp = complete_chat(chat_request)
        print(resp)
        
        if isinstance(resp, CompletionResponse):
            if resp.error:
                return jsonify({"error": f"Chat completion error: {resp.error}"}), 400
            return jsonify({"answer": resp.answer})
        elif resp is None:
            return jsonify({"error": "No response from chat completion"}), 500
        else:
            return jsonify({"error": "Unexpected response format"}), 500

    except Exception as e:
        app.logger.error(f"Unexpected error in completion_handler: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500

# Define a route for the '/analyst' endpoint that accepts POST requests
@app.route('/analyst', methods=['POST'])
def analyst():
    api_key = request.headers.get('X-API-KEY')
    validate_api_key(api_key)
    # Extract necessary parameters from the request JSON
    model_name = request.json.get('model_name')
    llm_type = request.json.get('llm_type')
    chat = request.json.get('chat')

    # Check if all required parameters are present
    if not model_name or not llm_type or not chat:
        return jsonify({"error": "Missing parameters"}), 400

    # Extract the data parameter from the request JSON
    data_param = request.json.get('data')
    if not data_param:
        return jsonify({"error": "Missing data parameter"}), 400

    # Read the data from the provided CSV file
    data = pd.read_csv(data_param)

    # Initialize the language model based on the provided type
    if llm_type == 'Groq':
        model_kwargs = {'seed': 26}
        llm = ChatGroq(model_name=model_name, temperature=0, api_key=os.environ['GROQ_API_KEY'], model_kwargs=model_kwargs)
    elif llm_type == 'Deepseek':
        llm = ChatOpenAI(model_name=model_name, temperature=0, seed=26, base_url='https://api.deepseek.com', api_key=os.environ['DEEPSEEK_API_KEY'])
    elif llm_type == 'Mistral':
        llm = ChatMistralAI(model_name=model_name, temperature=0, seed=26, api_key=os.environ['MISTRAL_API_KEY'])
    elif llm_type == 'OpenAI':
        llm = ChatOpenAI(model_name=model_name, temperature=0, seed=26, api_key=os.environ['OPENAI_API_KEY'])

    # Initialize the agent with the data and configuration
    agent = Agent(data, config={"llm": llm, "open_charts": False})

    # Perform the chat operation and get the response and explanation
    response = agent.chat(chat)
    explanation = agent.explain()

    # Convert the response to a DataFrame if it's a list
    if isinstance(response, list):
        try:
            response = pd.DataFrame(response)
        except Exception as e:
            return jsonify({"error": f"Failed to convert list to DataFrame: {str(e)}"}), 500

    # Convert the response to a dictionary
    if isinstance(response, pd.DataFrame):
        response_dict = response.to_dict(orient='records')
    elif isinstance(response, dict):
        response_dict = response
    else:
        response_dict = {'type': type(response).__name__, 'value': str(response)}

    return jsonify({"response": response_dict, "explanation": explanation})

# Define a route for the '/summarize' endpoint that returns a "not yet implemented" message
@app.route('/summarize', methods=['GET'])
def summarize():
    return "The /summarize endpoint is not yet implemented.", 501

# Define a route for the '/summarize' endpoint that returns a "not yet implemented" message
@app.route('/categorize', methods=['GET'])
def categorize():
    return "The /categorize endpoint is not yet implemented.", 501

# Define a route for the '/img-comparison' endpoint that returns a "not yet implemented" message
@app.route('/img-comparison', methods=['GET'])
def img_comparison():
    return "The /img-comparison endpoint is not yet implemented.", 501

# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    app.run(debug=True)
