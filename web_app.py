from flask import Flask, render_template, request, jsonify
from app import Retriever, CognitiveNeuroAssistant, KB_DOCS, FAQ

app = Flask(__name__)

# Initialize the assistant
retriever = Retriever(KB_DOCS, FAQ)
bot = CognitiveNeuroAssistant(retriever=retriever, mode="tutor")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
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

if __name__ == '__main__':
    app.run(debug=True, port=5000)
