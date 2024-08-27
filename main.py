from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
import pandas as pd
from pandasai import Agent

from langchain_groq.chat_models import ChatGroq
from langchain_openai import ChatOpenAI

app = Flask(__name__)

# Load environment variables
load_dotenv()

@app.route('/analyst', methods=['GET'])
def analyst():
    model_name = request.args.get('model_name')
    llm_type = request.args.get('llm_type')
    chat = request.args.get('chat')

    if not model_name or not llm_type or not chat:
        return jsonify({"error": "Missing parameters"}), 400

    data_param = request.args.get('data')
    if not data_param:
        return jsonify({"error": "Missing data parameter"}), 400

    # Dictionary to store the extracted dataframes
    data = pd.read_csv(data_param)

    if llm_type == 'Groq':
        model_kwargs = {'seed': 26}
        llm = ChatGroq(model_name=model_name, temperature=0, api_key=os.environ['GROQ_API_KEY'], model_kwargs=model_kwargs)

    elif llm_type == 'Deepseek':
        llm = ChatOpenAI(model_name=model_name, temperature=0, seed=26, api_key=os.environ['DEEPSEEK_API_KEY'])
    elif llm_type == 'Mistral':
        from langchain_mistralai import ChatMistral
        llm = ChatMistral(model_name=model_name, temperature=0, api_key=os.environ['MISTRAL_API_KEY'])

    agent = Agent(data, config={"llm": llm, "open_charts": False})

    response = agent.chat(chat)
    explanation = agent.explain()

    return jsonify({"response": response, "explanation": explanation})

if __name__ == '__main__':
    app.run(debug=True)
