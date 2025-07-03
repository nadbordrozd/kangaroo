// Contact Center AI Assistant - Frontend JavaScript

let conversationHistory = [];

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    console.log('DEBUG: Page loaded, initializing contact center assistant');
    
    // Add event listeners for Enter key on input fields
    document.getElementById('customerInput').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage('customer');
        }
    });
    
    document.getElementById('agentInput').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage('agent');
        }
    });
});

// Send message function
function sendMessage(sender) {
    const inputId = sender === 'customer' ? 'customerInput' : 'agentInput';
    const input = document.getElementById(inputId);
    const message = input.value.trim();
    
    if (!message) return;
    
    console.log(`DEBUG: Sending message - sender: ${sender}, message: ${message}`);
    
    // Add message to chat
    addMessageToChat(sender, message);
    
    // Add to conversation history
    conversationHistory.push({
        sender: sender,
        message: message,
        timestamp: new Date().toISOString()
    });
    
    // Clear input
    input.value = '';
    
    // Show loading indicator
    showLoadingIndicator();
    
    // Send to backend for AI analysis
    getAIAnalysis(message, sender);
}

// Add message to chat display
function addMessageToChat(sender, message) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    const timestamp = new Date().toLocaleTimeString();
    
    messageDiv.innerHTML = `
        <div class="message-header">
            <span class="sender">${sender.charAt(0).toUpperCase() + sender.slice(1)}</span>
            <span class="timestamp">${timestamp}</span>
        </div>
        <div class="message-content">${message}</div>
    `;
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Show loading indicator
function showLoadingIndicator() {
    const loadingIndicator = document.querySelector('.loading-indicator');
    if (loadingIndicator) {
        loadingIndicator.style.display = 'inline';
    }
}

// Hide loading indicator
function hideLoadingIndicator() {
    const loadingIndicator = document.querySelector('.loading-indicator');
    if (loadingIndicator) {
        loadingIndicator.style.display = 'none';
    }
}

// Get AI analysis from backend
async function getAIAnalysis(message, sender) {
    console.log(`DEBUG: Getting AI analysis for ${sender}: ${message}`);
    
    try {
        const requestData = {
            sender: sender,
            message: message,
            conversation_context: conversationHistory.slice(-10) // Send last 10 messages
        };
        
        console.log('DEBUG: Request data:', requestData);
        
        const response = await fetch('/send_message', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });
        
        console.log('DEBUG: Response status:', response.status);
        
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        
        const data = await response.json();
        console.log('DEBUG: Full response data:', data);
        
        if (data.success) {
            console.log('DEBUG: Suggestions received:', data.suggestions);
            updateAISuggestions(data.suggestions);
        } else {
            console.error('DEBUG: Response success=false');
            showError('Failed to get AI suggestions');
        }
        
    } catch (error) {
        console.error('DEBUG: Error in getAIAnalysis:', error);
        showError('Error connecting to AI service');
    } finally {
        hideLoadingIndicator();
    }
}

// Update AI suggestions panel - FIXED to work with backend response
function updateAISuggestions(suggestions) {
    console.log('DEBUG: updateAISuggestions called with:', suggestions);
    
    // Update summary with basic conversation info
    const summaryBox = document.getElementById('summaryBox');
    const conversationLength = conversationHistory.length;
    const latestMessage = conversationHistory[conversationLength - 1];
    
    summaryBox.innerHTML = `<p>📊 <strong>Conversation Summary:</strong> ${conversationLength} message(s) exchanged. Latest from ${latestMessage.sender}: "${latestMessage.message.substring(0, 50)}${latestMessage.message.length > 50 ? '...' : ''}"</p>`;
    
    // Update suggested responses - FIXED to handle array of strings
    const suggestionsList = document.getElementById('suggestionsList');
    if (suggestions && Array.isArray(suggestions) && suggestions.length > 0) {
        console.log('DEBUG: Displaying suggestions:', suggestions);
        suggestionsList.innerHTML = '';
        
        suggestions.forEach((suggestion, index) => {
            console.log(`DEBUG: Adding suggestion ${index}: ${suggestion}`);
            
            const suggestionDiv = document.createElement('div');
            suggestionDiv.className = 'suggestion-item';
            
            // Clean up quotes from the suggestion
            const cleanSuggestion = suggestion.replace(/^"|"$/g, '');
            
            suggestionDiv.innerHTML = `
                <div class="suggestion-content">
                    <p>${cleanSuggestion}</p>
                </div>
                <div class="suggestion-actions">
                    <button class="suggestion-btn use-btn" onclick="useSuggestion('${cleanSuggestion.replace(/'/g, "\\'")}')">
                        Use This Response
                    </button>
                    <button class="suggestion-btn copy-btn" onclick="copySuggestion('${cleanSuggestion.replace(/'/g, "\\'")}')">
                        Copy
                    </button>
                </div>
            `;
            suggestionsList.appendChild(suggestionDiv);
        });
        
        console.log('DEBUG: Suggestions displayed successfully');
    } else {
        console.log('DEBUG: No suggestions to display');
        suggestionsList.innerHTML = '<p class="placeholder-text">No suggested responses available.</p>';
    }
    
    // Update knowledge snippets with placeholder since we don't have this feature yet
    const knowledgeSnippets = document.getElementById('knowledgeSnippets');
    knowledgeSnippets.innerHTML = '<p class="placeholder-text">💡 Knowledge base integration will show relevant snippets here based on the conversation context.</p>';
}

// Use a suggested response
function useSuggestion(suggestion) {
    console.log('DEBUG: Using suggestion:', suggestion);
    const agentInput = document.getElementById('agentInput');
    agentInput.value = suggestion;
    agentInput.focus();
}

// Copy suggestion to clipboard
function copySuggestion(suggestion) {
    console.log('DEBUG: Copying suggestion:', suggestion);
    
    navigator.clipboard.writeText(suggestion).then(() => {
        // Show feedback
        const copyBtn = event.target;
        const originalText = copyBtn.textContent;
        copyBtn.textContent = 'Copied!';
        copyBtn.style.backgroundColor = '#27ae60';
        
        setTimeout(() => {
            copyBtn.textContent = originalText;
            copyBtn.style.backgroundColor = '';
        }, 1000);
    }).catch(err => {
        console.error('Failed to copy text: ', err);
        // Fallback for older browsers
        const textArea = document.createElement('textarea');
        textArea.value = suggestion;
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
    });
}

// Show error message
function showError(message) {
    console.log('DEBUG: Showing error:', message);
    const summaryBox = document.getElementById('summaryBox');
    summaryBox.innerHTML = `<p class="error-message">⚠️ ${message}</p>`;
}

// Clear conversation (optional feature)
function clearConversation() {
    if (confirm('Are you sure you want to clear the conversation?')) {
        conversationHistory = [];
        document.getElementById('chatMessages').innerHTML = '';
        
        // Reset AI panels to placeholder text
        document.getElementById('summaryBox').innerHTML = '<p class="placeholder-text">AI-generated summary will appear here based on the conversation and relevant SOPs.</p>';
        document.getElementById('suggestionsList').innerHTML = '<p class="placeholder-text">Suggested responses will appear here when there\'s an active conversation.</p>';
        document.getElementById('knowledgeSnippets').innerHTML = '<p class="placeholder-text">Relevant snippets from SOPs will appear here.</p>';
        
        console.log('DEBUG: Conversation cleared');
    }
}