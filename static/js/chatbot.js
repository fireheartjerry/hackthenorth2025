// Advanced Gemini-powered chatbot with live updates
class AppetizerIQChatbot {
    constructor() {
        this.isOpen = false;
        this.sessionId = this.generateSessionId();
        this.messageHistory = [];
        this.currentStream = null;
        this.pendingAction = null;
        
        this.initializeWidget();
        this.bindEvents();
        this.enableDragging();
        this.loadSuggestions();
        this.updateContext();
    }
    
    generateSessionId() {
        return 'chat_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    
    initializeWidget() {
        // Create chat widget HTML
        const chatWidget = document.createElement('div');
        chatWidget.id = 'chat-widget';
        chatWidget.className = 'chat-widget hidden';
        chatWidget.innerHTML = `
            <div class="chat-header">
                <div class="chat-header-content">
                    <div class="chat-avatar">🤖</div>
                    <div class="chat-title">
                        <h3>AppetizerIQ AI Assistant</h3>
                        <span class="chat-status" id="chat-status">Ready to help</span>
                    </div>
                </div>
                <div class="chat-controls">
                    <button id="chat-minimize" class="chat-btn" title="Minimize">−</button>
                    <button id="chat-close" class="chat-btn" title="Close">×</button>
                </div>
            </div>
            
            <div class="chat-body">
                <div class="chat-messages" id="chat-messages">
                    <div class="chat-message bot-message">
                        <div class="message-avatar">🤖</div>
                        <div class="message-content">
                            <div class="message-text">
                                Hi! I'm your AI assistant for the AppetizerIQ dashboard. I can help you:
                                <ul>
                                    <li>Filter and search submissions</li>
                                    <li>Switch between different modes</li>
                                    <li>Analyze portfolio data</li>
                                    <li>Explain underwriting decisions</li>
                                    <li>Navigate the dashboard</li>
                                </ul>
                                What would you like to do?
                            </div>
                            <div class="message-time">${new Date().toLocaleTimeString()}</div>
                        </div>
                    </div>
                </div>
                
                <div class="chat-input-container">
                    <div class="chat-suggestions" id="chat-suggestions">
                        <!-- Dynamic suggestions will be loaded here -->
                    </div>
                    
                    <div class="chat-input-area">
                        <textarea id="chat-input" placeholder="Ask me anything about your submissions..." rows="1"></textarea>
                        <button id="chat-send" class="chat-send-btn" disabled>
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <line x1="22" y1="2" x2="11" y2="13"></line>
                                <polygon points="22,2 15,22 11,13 2,9"></polygon>
                            </svg>
                        </button>
                    </div>
                </div>
            </div>
            
            <!-- Confirmation Modal -->
            <div class="chat-confirmation-modal hidden" id="chat-confirmation">
                <div class="confirmation-content">
                    <div class="confirmation-header">
                        <h4>Confirm Action</h4>
                    </div>
                    <div class="confirmation-body">
                        <p id="confirmation-message"></p>
                    </div>
                    <div class="confirmation-actions">
                        <button id="confirm-yes" class="btn btn-primary">Yes, do it</button>
                        <button id="confirm-no" class="btn btn-secondary">Cancel</button>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(chatWidget);
        
        // Create floating chat button
        const chatButton = document.createElement('button');
        chatButton.id = 'chat-toggle';
        chatButton.className = 'chat-toggle-btn';
        chatButton.innerHTML = `
            <div class="chat-toggle-icon">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
                    <circle cx="9" cy="10" r="1"/>
                    <circle cx="13" cy="10" r="1"/>
                    <circle cx="17" cy="10" r="1"/>
                </svg>
            </div>
            <div class="chat-toggle-badge hidden" id="chat-badge">1</div>
        `;
        
        document.body.appendChild(chatButton);
        
        // Add CSS
        this.addStyles();
    }
    
    addStyles() {
        const styles = `
            .chat-toggle-btn {
                position: fixed;
                bottom: 20px;
                right: 20px;
                width: 56px;
                height: 56px;
                border-radius: 50%;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border: none;
                color: white;
                cursor: pointer;
                box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
                z-index: 1000;
                transition: all 0.3s ease;
                display: flex;
                align-items: center;
                justify-content: center;
                position: relative;
                padding: 0;
            }
            
            .chat-toggle-btn:hover {
                transform: scale(1.1);
                box-shadow: 0 6px 25px rgba(102, 126, 234, 0.6);
            }
            
            .chat-toggle-badge {
                position: absolute;
                top: -5px;
                right: -5px;
                background: #ef4444;
                color: white;
                border-radius: 50%;
                width: 20px;
                height: 20px;
                font-size: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
            }
            
            .chat-widget {
                position: fixed;
                bottom: 90px;
                right: 20px;
                width: 400px;
                height: 600px;
                background: rgba(15, 23, 42, 0.95);
                backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 12px;
                overflow: hidden;
                z-index: 999;
                display: flex;
                flex-direction: column;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
                transition: all 0.3s ease;
            }
            
            .chat-widget.minimized {
                height: 60px;
            }
            
            .chat-header {
                padding: 16px;
                border-bottom: 1px solid rgba(148, 163, 184, 0.2);
                display: flex;
                align-items: center;
                justify-content: space-between;
                background: rgba(30, 41, 59, 0.8);
                user-select: none;
                cursor: move;
            }
            
            .chat-header-content {
                display: flex;
                align-items: center;
                gap: 12px;
            }
            
            .chat-avatar {
                width: 40px;
                height: 40px;
                border-radius: 50%;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 20px;
            }
            
            .chat-title h3 {
                margin: 0;
                font-size: 16px;
                font-weight: 600;
                color: white;
            }
            
            .chat-status {
                font-size: 12px;
                color: #94a3b8;
            }
            
            .chat-controls {
                display: flex;
                gap: 8px;
            }
            
            .chat-btn {
                width: 32px;
                height: 32px;
                border: none;
                background: rgba(148, 163, 184, 0.2);
                color: white;
                border-radius: 8px;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: background 0.2s;
            }
            
            .chat-btn:hover {
                background: rgba(148, 163, 184, 0.3);
            }
            
            .chat-body {
                flex: 1;
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }
            
            .chat-messages {
                flex: 1;
                padding: 16px;
                overflow-y: auto;
                display: flex;
                flex-direction: column;
                gap: 16px;
            }
            
            .chat-message {
                display: flex;
                gap: 12px;
                animation: slideIn 0.3s ease;
            }
            
            .chat-message.user-message {
                flex-direction: row-reverse;
            }
            
            .message-avatar {
                width: 32px;
                height: 32px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 16px;
                flex-shrink: 0;
            }
            
            .bot-message .message-avatar {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }
            
            .user-message .message-avatar {
                background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            }
            
            .message-content {
                max-width: 280px;
                background: rgba(30, 41, 59, 0.8);
                border-radius: 12px;
                padding: 12px 16px;
                border: 1px solid rgba(148, 163, 184, 0.2);
            }
            
            .user-message .message-content {
                background: rgba(16, 185, 129, 0.2);
                border-color: rgba(16, 185, 129, 0.3);
            }
            
            .message-text {
                color: white;
                font-size: 14px;
                line-height: 1.5;
                margin-bottom: 8px;
            }
            
            .message-text ul {
                margin: 8px 0;
                padding-left: 20px;
            }
            
            .message-text li {
                margin: 4px 0;
            }
            
            .message-time {
                font-size: 11px;
                color: #64748b;
            }
            
            .typing-indicator {
                display: flex;
                gap: 4px;
                padding: 8px 0;
            }
            
            .typing-dot {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: #667eea;
                animation: typing 1.4s infinite ease-in-out;
            }
            
            .typing-dot:nth-child(1) { animation-delay: -0.32s; }
            .typing-dot:nth-child(2) { animation-delay: -0.16s; }
            
            .chat-suggestions {
                padding: 12px 16px;
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                border-bottom: 1px solid rgba(148, 163, 184, 0.2);
            }
            
            .suggestion-btn {
                padding: 6px 12px;
                background: rgba(102, 126, 234, 0.2);
                border: 1px solid rgba(102, 126, 234, 0.3);
                border-radius: 20px;
                color: #a5b4fc;
                font-size: 12px;
                cursor: pointer;
                transition: all 0.2s;
                display: flex;
                align-items: center;
                gap: 6px;
            }
            
            .suggestion-icon {
                font-size: 14px;
            }
            
            .suggestion-btn:hover {
                background: rgba(102, 126, 234, 0.3);
                color: white;
            }
            
            .chat-input-container {
                border-top: 1px solid rgba(148, 163, 184, 0.2);
                background: rgba(15, 23, 42, 0.9);
            }

            .chat-input-area {
                padding: 12px 16px;
                display: flex;
                gap: 12px;
                align-items: flex-end;
            }
            
            #chat-input {
                flex: 1;
                background: rgba(30, 41, 59, 0.8);
                border: 1px solid rgba(148, 163, 184, 0.2);
                border-radius: 12px;
                padding: 12px 16px;
                color: white;
                font-size: 14px;
                resize: none;
                max-height: 120px;
                min-height: 44px;
            }
            
            #chat-input:focus {
                outline: none;
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            }
            
            .chat-send-btn {
                width: 44px;
                height: 44px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border: none;
                border-radius: 12px;
                color: white;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.2s;
            }
            
            .chat-send-btn:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }
            
            .chat-send-btn:not(:disabled):hover {
                transform: scale(1.05);
            }
            
            .chat-confirmation-modal {
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0, 0, 0, 0.8);
                display: flex;
                align-items: center;
                justify-content: center;
                border-radius: 16px;
            }
            
            .confirmation-content {
                background: rgba(15, 23, 42, 0.95);
                border: 1px solid rgba(148, 163, 184, 0.2);
                border-radius: 12px;
                padding: 24px;
                max-width: 320px;
                text-align: center;
            }
            
            .confirmation-header h4 {
                margin: 0 0 16px 0;
                color: white;
                font-size: 18px;
            }
            
            .confirmation-body p {
                margin: 0 0 24px 0;
                color: #cbd5e1;
                line-height: 1.5;
            }
            
            .confirmation-actions {
                display: flex;
                gap: 12px;
                justify-content: center;
            }
            
            .btn {
                padding: 10px 20px;
                border: none;
                border-radius: 8px;
                font-weight: 500;
                cursor: pointer;
                transition: all 0.2s;
            }
            
            .btn-primary {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            
            .btn-secondary {
                background: rgba(148, 163, 184, 0.2);
                color: white;
                border: 1px solid rgba(148, 163, 184, 0.3);
            }
            
            .btn:hover {
                transform: translateY(-1px);
            }
            
            .action-preview {
                background: rgba(16, 185, 129, 0.1);
                border: 1px solid rgba(16, 185, 129, 0.3);
                border-radius: 8px;
                padding: 12px;
                margin: 12px 0;
            }
            
            .action-preview-title {
                font-weight: 600;
                color: #10b981;
                margin-bottom: 8px;
            }
            
            .action-preview-details {
                font-size: 13px;
                color: #cbd5e1;
            }
            
            @keyframes slideIn {
                from {
                    opacity: 0;
                    transform: translateY(20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            @keyframes typing {
                0%, 80%, 100% {
                    transform: scale(0);
                }
                40% {
                    transform: scale(1);
                }
            }
            
            .hidden {
                display: none !important;
            }
            
            @media (max-width: 480px) {
                .chat-widget {
                    width: calc(100vw - 40px);
                    height: calc(100vh - 140px);
                    bottom: 90px;
                    right: 20px;
                    left: 20px;
                }
            }
        `;
        
        const styleSheet = document.createElement('style');
        styleSheet.textContent = styles;
        document.head.appendChild(styleSheet);
    }
    
    bindEvents() {
        // Toggle chat widget
        document.getElementById('chat-toggle').addEventListener('click', () => {
            this.toggleWidget();
        });
        
        // Close and minimize buttons
        document.getElementById('chat-close').addEventListener('click', () => {
            this.closeWidget();
        });
        
        document.getElementById('chat-minimize').addEventListener('click', () => {
            this.minimizeWidget();
        });
        
        // Send message
        document.getElementById('chat-send').addEventListener('click', () => {
            this.sendMessage();
        });
        
        // Input handling
        const chatInput = document.getElementById('chat-input');
        chatInput.addEventListener('input', () => {
            this.handleInputChange();
        });
        
        chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Suggestion buttons
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('suggestion-btn')) {
                const message = e.target.dataset.message;
                this.sendMessage(message);
            }
        });
        
        // Confirmation modal
        document.getElementById('confirm-yes').addEventListener('click', () => {
            this.executeAction();
        });
        
        document.getElementById('confirm-no').addEventListener('click', () => {
            this.hideConfirmation();
        });
    }

    enableDragging() {
        const widget = document.getElementById('chat-widget');
        const header = widget.querySelector('.chat-header');
        let isDragging = false;
        let startX, startY, startLeft, startTop;

        header.addEventListener('mousedown', (e) => {
            isDragging = true;
            const rect = widget.getBoundingClientRect();
            startX = e.clientX;
            startY = e.clientY;
            startLeft = rect.left;
            startTop = rect.top;
            widget.style.bottom = 'auto';
            widget.style.right = 'auto';
            widget.style.left = `${startLeft}px`;
            widget.style.top = `${startTop}px`;
            document.addEventListener('mousemove', onMouseMove);
            document.addEventListener('mouseup', onMouseUp);
        });

        const onMouseMove = (e) => {
            if (!isDragging) return;
            const dx = e.clientX - startX;
            const dy = e.clientY - startY;
            widget.style.left = `${startLeft + dx}px`;
            widget.style.top = `${startTop + dy}px`;
        };

        const onMouseUp = () => {
            isDragging = false;
            document.removeEventListener('mousemove', onMouseMove);
            document.removeEventListener('mouseup', onMouseUp);
        };
    }
    
    toggleWidget() {
        const widget = document.getElementById('chat-widget');
        const badge = document.getElementById('chat-badge');
        const toggleBtn = document.getElementById('chat-toggle');

        if (this.isOpen) {
            this.closeWidget();
        } else {
            widget.classList.remove('hidden');
            toggleBtn.classList.add('hidden');
            this.isOpen = true;
            badge.classList.add('hidden');
            document.getElementById('chat-input').focus();
        }
    }

    closeWidget() {
        const widget = document.getElementById('chat-widget');
        const toggleBtn = document.getElementById('chat-toggle');
        widget.classList.add('hidden');
        toggleBtn.classList.remove('hidden');
        this.isOpen = false;
    }
    
    minimizeWidget() {
        const widget = document.getElementById('chat-widget');
        widget.classList.toggle('minimized');
    }
    
    handleInputChange() {
        const input = document.getElementById('chat-input');
        const sendBtn = document.getElementById('chat-send');
        
        sendBtn.disabled = !input.value.trim();
        
        // Auto-resize textarea
        input.style.height = 'auto';
        input.style.height = Math.min(input.scrollHeight, 120) + 'px';
    }
    
    async sendMessage(message = null) {
        const input = document.getElementById('chat-input');
        const messageText = message || input.value.trim();
        
        if (!messageText) return;
        
        // Clear input
        input.value = '';
        this.handleInputChange();
        
        // Add user message to chat
        this.addMessage(messageText, 'user');
        
        // Show typing indicator
        this.showTyping();
        
        try {
            // Use streaming for better UX
            await this.streamResponse(messageText);
        } catch (error) {
            this.hideTyping();
            this.addMessage('Sorry, I encountered an error. Please try again.', 'bot');
            console.error('Chat error:', error);
        }
    }
    
    async streamResponse(message) {
        const messagesContainer = document.getElementById('chat-messages');
        
        // Create bot message container
        const botMessageDiv = document.createElement('div');
        botMessageDiv.className = 'chat-message bot-message';
        botMessageDiv.innerHTML = `
            <div class="message-avatar">🤖</div>
            <div class="message-content">
                <div class="message-text streaming-text"></div>
                <div class="message-time">${new Date().toLocaleTimeString()}</div>
            </div>
        `;

        this.hideTyping();
        messagesContainer.appendChild(botMessageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;

        const streamingText = botMessageDiv.querySelector('.streaming-text');
        
        try {
            const url = `/api/chat/stream?message=${encodeURIComponent(message)}&session_id=${this.sessionId}`;
            const eventSource = new EventSource(url);
            
            let accumulatedText = '';
            
            eventSource.onmessage = (event) => {
                const data = JSON.parse(event.data);
                
                if (data.type === 'chunk') {
                    accumulatedText = data.accumulated;
                    streamingText.innerHTML = this.formatMessage(accumulatedText);
                    messagesContainer.scrollTop = messagesContainer.scrollHeight;
                }
                else if (data.type === 'action') {
                    this.pendingAction = data.content;
                    this.showConfirmation(data.confirmation);
                }
                else if (data.type === 'end') {
                    streamingText.innerHTML = this.formatMessage(data.content);
                    eventSource.close();
                }
                else if (data.type === 'error') {
                    streamingText.innerHTML = `Error: ${data.content}`;
                    eventSource.close();
                }
            };
            
            eventSource.onerror = () => {
                streamingText.innerHTML = 'Connection error. Please try again.';
                eventSource.close();
            };
            
        } catch (error) {
            streamingText.innerHTML = 'Failed to connect. Please try again.';
        }
    }
    
    formatMessage(text) {
        // Basic markdown-like formatting
        return text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>');
    }
    
    addMessage(text, sender) {
        const messagesContainer = document.getElementById('chat-messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${sender}-message`;
        
        const avatar = sender === 'user' ? '👤' : '🤖';
        
        messageDiv.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                <div class="message-text">${this.formatMessage(text)}</div>
                <div class="message-time">${new Date().toLocaleTimeString()}</div>
            </div>
        `;
        
        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        
        this.messageHistory.push({ text, sender, timestamp: new Date() });
    }
    
    showTyping() {
        const messagesContainer = document.getElementById('chat-messages');
        const typingDiv = document.createElement('div');
        typingDiv.className = 'chat-message bot-message';
        typingDiv.id = 'typing-indicator';
        typingDiv.innerHTML = `
            <div class="message-avatar">🤖</div>
            <div class="message-content">
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        `;
        
        messagesContainer.appendChild(typingDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
    
    hideTyping() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }
    
    showConfirmation(message) {
        const modal = document.getElementById('chat-confirmation');
        const messageEl = document.getElementById('confirmation-message');
        
        messageEl.textContent = message;
        modal.classList.remove('hidden');
    }
    
    hideConfirmation() {
        const modal = document.getElementById('chat-confirmation');
        modal.classList.add('hidden');
        this.pendingAction = null;
    }
    
    async executeAction() {
        if (!this.pendingAction) return;
        
        try {
            const response = await fetch('/api/chat/execute', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    action: this.pendingAction
                })
            });
            
            const result = await response.json();
            
            if (result.ok) {
                // Apply the action to the current page
                this.applyAction(result);
                this.addMessage(`✅ ${result.message}`, 'bot');
            } else {
                this.addMessage(`❌ Failed to execute action: ${result.error}`, 'bot');
            }
        } catch (error) {
            this.addMessage('❌ Failed to execute action. Please try again.', 'bot');
            console.error('Action execution error:', error);
        }
        
        this.hideConfirmation();
    }
    
    applyAction(result) {
        const { action_type, url_params } = result;
        
        // Update the current page based on the action
        if (action_type === 'filter' || action_type === 'mode' || action_type === 'search' || action_type === 'sort') {
            // Update form fields if they exist
            const params = new URLSearchParams(url_params);
            
            for (const [key, value] of params) {
                const element = document.getElementById(`f_${key}`) || document.getElementById(key);
                if (element) {
                    element.value = value;
                }
            }
            
            // Trigger refresh if we're on submissions page
            if (window.location.pathname.includes('submissions') || window.location.pathname === '/') {
                const applyBtn = document.getElementById('btn_apply');
                if (applyBtn) {
                    applyBtn.click();
                } else {
                    // Fallback: reload with new params
                    const currentUrl = new URL(window.location);
                    for (const [key, value] of params) {
                        currentUrl.searchParams.set(key, value);
                    }
                    window.location.href = currentUrl.toString();
                }
            }
            
            // Update localStorage for mode changes
            if (action_type === 'mode') {
                const mode = params.get('mode');
                if (mode) {
                    localStorage.setItem('ACTIVE_MODE', mode);
                    // Trigger storage event for other components
                    window.dispatchEvent(new StorageEvent('storage', {
                        key: 'ACTIVE_MODE',
                        newValue: mode
                    }));
                }
            }
        }
    }
    
    updateStatus(status) {
        const statusEl = document.getElementById('chat-status');
        if (statusEl) {
            statusEl.textContent = status;
        }
    }
    
    showNotification() {
        const badge = document.getElementById('chat-badge');
        if (!this.isOpen) {
            badge.classList.remove('hidden');
        }
    }
    
    async loadSuggestions() {
        try {
            const response = await fetch('/api/chat/suggestions');
            const data = await response.json();
            
            if (data.suggestions && data.suggestions.length > 0) {
                this.updateSuggestions(data.suggestions);
            }
        } catch (error) {
            console.warn('Failed to load chat suggestions:', error);
            // Fallback to default suggestions
            this.updateSuggestions([
                { text: "Show me target submissions in CA", icon: "🎯" },
                { text: "Filter by premium over $100k", icon: "💰" },
                { text: "Switch to unicorn mode", icon: "🦄" },
                { text: "Explain the current filters", icon: "💡" }
            ]);
        }
    }
    
    updateSuggestions(suggestions) {
        const container = document.getElementById('chat-suggestions');
        if (!container) return;
        
        container.innerHTML = suggestions.map(suggestion => 
            `<button class="suggestion-btn" data-message="${suggestion.text}">
                <span class="suggestion-icon">${suggestion.icon || '💬'}</span>
                ${suggestion.text}
            </button>`
        ).join('');
    }
    
    async updateContext() {
        try {
            // Get current dashboard state
            const currentMode = localStorage.getItem('ACTIVE_MODE') || 'balanced';
            const currentPage = window.location.pathname;
            
            // Get visible submission count if on submissions page
            let submissionCount = 0;
            const submissionRows = document.querySelectorAll('#tbody tr, .submission-row');
            if (submissionRows.length > 0) {
                submissionCount = submissionRows.length;
            }
            
            // Get active filters
            const filters = {};
            const filterInputs = document.querySelectorAll('[id^="f_"]');
            filterInputs.forEach(input => {
                if (input.value) {
                    const key = input.id.replace('f_', '');
                    filters[key] = input.value;
                }
            });
            
            // Send context to backend
            await fetch('/api/chat/context', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    mode: currentMode,
                    page: currentPage,
                    submission_count: submissionCount,
                    filters: filters,
                    last_action: this.lastAction || ''
                })
            });
            
        } catch (error) {
            console.warn('Failed to update chat context:', error);
        }
    }
    
    setLastAction(action) {
        this.lastAction = action;
        this.updateContext();
    }
}

// Initialize chatbot when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.AppetizerIQChatbot = new AppetizerIQChatbot();
});

// Export for use in other scripts
window.AppetizerIQChatbot = AppetizerIQChatbot;