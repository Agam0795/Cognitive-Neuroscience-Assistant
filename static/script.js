// Get DOM elements
const chatContainer = document.getElementById('chatContainer');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const btnText = document.getElementById('btnText');
const btnLoader = document.getElementById('btnLoader');
const modeSelect = document.getElementById('mode');

// Send message function
async function sendMessage() {
    const message = userInput.value.trim();
    
    if (!message) {
        return;
    }
    
    // Add user message to chat
    addMessage(message, 'user');
    
    // Clear input
    userInput.value = '';
    
    // Disable send button and show loader
    setLoading(true);
    
    try {
        // Send request to backend
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        });
        
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        
        const data = await response.json();
        
        // Add bot response to chat
        addMessage(data.response, 'bot');
        
        // Update mode if changed
        if (data.mode) {
            modeSelect.value = data.mode;
        }
        
    } catch (error) {
        console.error('Error:', error);
        addMessage('Sorry, there was an error processing your request. Please try again.', 'bot');
    } finally {
        setLoading(false);
    }
}

// Add message to chat container
function addMessage(text, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    if (sender === 'user') {
        contentDiv.innerHTML = `<strong>You:</strong> ${escapeHtml(text)}`;
    } else {
        // Format bot message (preserve line breaks and markdown-style formatting)
        const formattedText = formatBotMessage(text);
        contentDiv.innerHTML = `<strong>Assistant:</strong> ${formattedText}`;
    }
    
    messageDiv.appendChild(contentDiv);
    chatContainer.appendChild(messageDiv);
    
    // Scroll to bottom
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Format bot message with basic markdown support
function formatBotMessage(text) {
    // Escape HTML first
    let formatted = escapeHtml(text);
    
    // Convert line breaks
    formatted = formatted.replace(/\n\n/g, '<br><br>');
    formatted = formatted.replace(/\n/g, '<br>');
    
    // Convert markdown-style bold *text*
    formatted = formatted.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    
    // Convert markdown-style code `code`
    formatted = formatted.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    return formatted;
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}

// Set loading state
function setLoading(isLoading) {
    sendBtn.disabled = isLoading;
    if (isLoading) {
        btnText.classList.add('hidden');
        btnLoader.classList.remove('hidden');
    } else {
        btnText.classList.remove('hidden');
        btnLoader.classList.add('hidden');
    }
}

// Change mode
async function changeMode() {
    const mode = modeSelect.value;
    
    try {
        const response = await fetch('/mode', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ mode: mode })
        });
        
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        
        const data = await response.json();
        
        // Show confirmation message
        addMessage(`Mode changed to ${data.mode}`, 'bot');
        
    } catch (error) {
        console.error('Error:', error);
        addMessage('Sorry, there was an error changing the mode.', 'bot');
    }
}

// Set question from example tag
function setQuestion(question) {
    userInput.value = question;
    userInput.focus();
}

// Handle Enter key press (Shift+Enter for new line)
function handleKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

// Focus input on load
window.addEventListener('load', () => {
    userInput.focus();
});
