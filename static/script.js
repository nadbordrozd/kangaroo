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
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault(); // Prevent new line
            sendMessage('agent');
        }
        // Allow Shift+Enter for new lines in textarea
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
    
    // Only process customer messages with AI analysis
    if (sender === 'customer') {
        // Show loading indicator
        showLoadingIndicator();
        
        // Send to backend for AI analysis
        getAIAnalysis(message, sender);
    }
    // For agent messages: do absolutely nothing - leave AI panel unchanged
}

// Add message to chat display
function addMessageToChat(sender, message) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    const timestamp = new Date().toLocaleTimeString();
    
    // Convert line breaks to HTML breaks for proper display
    const formattedMessage = message.replace(/\n/g, '<br>');
    
    messageDiv.innerHTML = `
        <div class="message-header">
            <span class="sender">${sender.charAt(0).toUpperCase() + sender.slice(1)}</span>
            <span class="timestamp">${timestamp}</span>
        </div>
        <div class="message-content">${formattedMessage}</div>
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
            console.log('DEBUG: Knowledge snippets received:', data.knowledge_snippets);
            updateAISuggestions(data.suggestions, data.knowledge_snippets);
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
function updateAISuggestions(suggestions, knowledgeSnippets = []) {
    console.log('DEBUG: updateAISuggestions called with:', suggestions);
    console.log('DEBUG: updateAISuggestions knowledge snippets:', knowledgeSnippets);
    
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
            
            // Clean up quotes from the suggestion and convert line breaks
            const cleanSuggestion = suggestion.replace(/^"|"$/g, '');
            const formattedSuggestion = cleanSuggestion.replace(/\n/g, '<br>');
            
            suggestionDiv.innerHTML = `
                <div class="suggestion-content">
                    <p>${formattedSuggestion}</p>
                </div>
                <div class="suggestion-actions">
                    <button class="suggestion-btn use-btn">
                        Use This Response
                    </button>
                </div>
            `;
            
            // Add event listener to avoid inline onclick issues with special characters
            const useButton = suggestionDiv.querySelector('.use-btn');
            useButton.addEventListener('click', () => useSuggestion(cleanSuggestion));
            
            suggestionsList.appendChild(suggestionDiv);
        });
        
        console.log('DEBUG: Suggestions displayed successfully');
    } else {
        console.log('DEBUG: No suggestions to display');
        suggestionsList.innerHTML = '<p class="placeholder-text">No suggested responses available.</p>';
    }
    
    // Update knowledge snippets
    const knowledgeSnippetsElement = document.getElementById('knowledgeSnippets');
    if (knowledgeSnippets && Array.isArray(knowledgeSnippets) && knowledgeSnippets.length > 0) {
        console.log('DEBUG: Displaying knowledge snippets:', knowledgeSnippets);
        knowledgeSnippetsElement.innerHTML = '';
        
        knowledgeSnippets.forEach((snippet, index) => {
            console.log(`DEBUG: Adding knowledge snippet ${index}: ${snippet.length} characters`);
            
            const snippetDiv = document.createElement('div');
            snippetDiv.className = 'knowledge-snippet';
            
            // Convert markdown-style formatting to HTML
            const htmlSnippet = snippet
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') // Bold text
                .replace(/\n/g, '<br>'); // Line breaks
            
            snippetDiv.innerHTML = `
                <div class="snippet-content">
                    <p>${htmlSnippet}</p>
                </div>
            `;
            knowledgeSnippetsElement.appendChild(snippetDiv);
        });
        
        console.log('DEBUG: Knowledge snippets displayed successfully');
    } else {
        console.log('DEBUG: No knowledge snippets to display');
        knowledgeSnippetsElement.innerHTML = '<p class="placeholder-text">💡 No relevant knowledge base snippets found for this customer inquiry.</p>';
    }
}


// Use a suggested response
function useSuggestion(suggestion) {
    console.log('DEBUG: Using suggestion:', suggestion);
    const agentInput = document.getElementById('agentInput');
    agentInput.value = suggestion;
    agentInput.focus();
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