document.getElementById("user-input").addEventListener("submit", function(event) {
    event.preventDefault();
    
    var userInput = document.getElementById("user-input-text").value.trim();
    
    if (userInput === '') {
        return;
    }
    
    // Create user message element
    var userMessageElement = document.createElement("div");
    userMessageElement.className = "message user-message";
    userMessageElement.textContent = userInput;
    
    // Append user message to chat container
    var chatContainer = document.getElementById("chat-container");
    chatContainer.appendChild(userMessageElement);
    
    // Clear input field
    document.getElementById("user-input-text").value = "";
    
    // Send user input to server
    fetch('/chatbot', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: 'user_input=' + encodeURIComponent(userInput),
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        // Handle bot response
        if (data.error) {
            console.error(data.error);
            return;
        }
        
        var botResponse = data.bot_response;
        
        // Create bot message element
        var botMessageElement = document.createElement("div");
        botMessageElement.className = "message bot-message";
        botMessageElement.innerHTML = botResponse; // Use innerHTML to render HTML content
        
        // Append bot message to chat container
        chatContainer.appendChild(botMessageElement);
        
        // Scroll to bottom of chat container
        chatContainer.scrollTop = chatContainer.scrollHeight;
    })
    .catch(error => {
        console.error('Error:', error);
    });
});
