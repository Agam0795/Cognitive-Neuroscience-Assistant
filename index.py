from flask import Flask, render_template, request, jsonify
import os
import sys

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(__file__))

from app import Retriever, CognitiveNeuroAssistant, KB_DOCS, FAQ

app = Flask(__name__)

# Initialize the assistant (singleton pattern for serverless)
_retriever = None
_bot = None

def get_bot():
    global _retriever, _bot
    if _bot is None:
        _retriever = Retriever(KB_DOCS, FAQ)
        _bot = CognitiveNeuroAssistant(retriever=_retriever, mode="tutor")
    return _bot

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        bot = get_bot()
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        # Get response from the assistant
        response = bot.answer(user_message)
        
        return jsonify({
            'response': response,
            'mode': bot.mode
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/mode', methods=['POST'])
def change_mode():
    try:
        bot = get_bot()
        data = request.get_json()
        mode = data.get('mode', 'tutor').strip().lower()
        
        if mode not in ['tutor', 'concise']:
            return jsonify({'error': 'Invalid mode'}), 400
        
        bot.set_mode(mode)
        
        return jsonify({
            'mode': bot.mode,
            'message': f'Mode changed to {bot.mode}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# For local development
if __name__ == '__main__':
    app.run(debug=True, port=5000)
