import { useState, useRef, useEffect } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faComment, faPaperPlane, faTimes, faCode } from '@fortawesome/free-solid-svg-icons';

export default function Chat() {
    const [messages, setMessages] = useState([
        {
            id: 1,
            type: 'ai',
            content: "Hello! I'm your Tech Support AI Assistant. How can I help you today?"
        }
    ]);
    const [newMessage, setNewMessage] = useState('');
    const [isTyping, setIsTyping] = useState(false);
    const [isChatOpen, setIsChatOpen] = useState(false);
    const [showDebug, setShowDebug] = useState(false);
    const [debugInfo, setDebugInfo] = useState(null);
    const chatContainerRef = useRef(null);

    const scrollToBottom = () => {
        if (chatContainerRef.current) {
            chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
        }
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const sendMessage = async (e) => {
        e.preventDefault();
        if (!newMessage.trim()) return;

        // Add user message
        setMessages(prev => [...prev, {
            id: Date.now(),
            type: 'user',
            content: newMessage
        }]);

        const userMessage = newMessage;
        setNewMessage('');
        setIsTyping(true);
        setDebugInfo(null);

        try {
            const response = await fetch('http://127.0.0.1:8000/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                credentials: 'include',
                body: JSON.stringify({ query: userMessage })
            });

            const data = await response.json();

            // Create debug info
            let debugText = '';
            if (data.source === 'database') {
                debugText = `Source: Database Guide\nSimilarity Score: ${(data.similarity * 100).toFixed(1)}%`;
            } else if (data.source === 'ai_database') {
                debugText = `Source: Learned AI Response\nSimilarity Score: ${(data.similarity * 100).toFixed(1)}%`;
            } else if (data.source === 'gemini') {
                debugText = `Source: New Gemini Response\nDatabase Context Used: ${data.context_used ? 'Yes' : 'No'}\nNumber of Relevant Contexts: ${data.num_contexts}`;
            }
            setDebugInfo(debugText);

            if (data.error) {
                setMessages(prev => [...prev, {
                    id: Date.now(),
                    type: 'ai',
                    content: data.error
                }]);
            } else {
                setMessages(prev => [...prev, {
                    id: Date.now(),
                    type: 'ai',
                    content: data
                }]);
            }
        } catch (error) {
            console.error('Error:', error);
            setMessages(prev => [...prev, {
                id: Date.now(),
                type: 'ai',
                content: 'Sorry, I encountered an error. Please try again.'
            }]);
        } finally {
            setIsTyping(false);
        }
    };

    return (
        <>
            {/* Floating Chat Button - Only show when chat is closed */}
            {!isChatOpen && (
                <button
                    onClick={() => setIsChatOpen(true)}
                    className="fixed bottom-6 right-6 w-14 h-14 bg-blue-500 rounded-full shadow-lg 
                             hover:bg-blue-600 focus:outline-none
                             flex items-center justify-center text-white z-50 transition-all
                             hover:scale-110"
                    aria-label="Open chat"
                >
                    <FontAwesomeIcon icon={faComment} className="text-xl" />
                </button>
            )}

            {/* Floating Chat Window */}
            <div className={`fixed transition-all duration-300 transform z-50
                         md:bottom-6 md:right-6 md:w-96
                         ${isChatOpen 
                             ? 'inset-0 md:inset-auto translate-y-0 opacity-100' 
                             : 'translate-y-8 opacity-0 pointer-events-none'}`}
            >
                <div className="bg-[#1e2635] flex flex-col overflow-hidden h-full md:h-[600px] md:rounded-lg md:shadow-xl">
                    {/* Chat Header */}
                    <div className="bg-blue-500 text-white px-4 py-4 flex justify-between items-center">
                        <h2 className="text-lg font-medium flex-1 text-center">Tech Support AI Assistant</h2>
                        <div className="flex items-center gap-2 absolute right-2">
                            <button 
                                onClick={() => setShowDebug(!showDebug)}
                                className="text-white hover:text-gray-200 focus:outline-none
                                         w-8 h-8 flex items-center justify-center rounded-full
                                         hover:bg-blue-600/50 transition-colors"
                                aria-label="Toggle debug info"
                            >
                                <FontAwesomeIcon icon={faCode} className="text-sm" />
                            </button>
                            <button 
                                onClick={() => setIsChatOpen(false)}
                                className="text-white hover:text-gray-200 focus:outline-none
                                         w-8 h-8 flex items-center justify-center rounded-full
                                         hover:bg-blue-600/50 transition-colors"
                                aria-label="Close chat"
                            >
                                <FontAwesomeIcon icon={faTimes} className="text-lg" />
                            </button>
                        </div>
                    </div>

                    {/* Debug Info */}
                    {showDebug && debugInfo && (
                        <div className="bg-gray-900 text-gray-300 p-2 text-xs font-mono">
                            {debugInfo.split('\n').map((line, i) => (
                                <div key={i}>{line}</div>
                            ))}
                        </div>
                    )}

                    {/* Chat Messages */}
                    <div 
                        ref={chatContainerRef}
                        className="flex-1 overflow-y-auto p-4 space-y-3 custom-scrollbar bg-[#1e2635]"
                    >
                        {messages.map(message => (
                            <div 
                                key={message.id}
                                className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                            >
                                <div className={`
                                    max-w-[85%] rounded-lg py-2.5 px-4
                                    ${message.type === 'user' 
                                        ? 'bg-blue-500 text-white' 
                                        : 'bg-[#2a3343] text-gray-100'}
                                `}>
                                    {message.type === 'ai' && message.content.issue ? (
                                        <>
                                            <div className="font-semibold">{message.content.issue}</div>
                                            <div className="mt-1 text-gray-200">{message.content.solution}</div>
                                        </>
                                    ) : (
                                        <div className="text-sm">{message.content}</div>
                                    )}
                                </div>
                            </div>
                        ))}

                        {/* Typing Indicator */}
                        {isTyping && (
                            <div className="flex justify-start">
                                <div className="bg-[#2a3343] rounded-lg py-2 px-4">
                                    <div className="flex space-x-2">
                                        {[0, 1, 2].map((i) => (
                                            <div 
                                                key={i}
                                                className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce"
                                                style={{ animationDelay: `${i * 0.2}s` }}
                                            />
                                        ))}
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Input Area */}
                    <div className="p-4 bg-[#1e2635] border-t border-gray-800">
                        <form onSubmit={sendMessage} className="flex items-center space-x-2">
                            <input 
                                value={newMessage}
                                onChange={(e) => setNewMessage(e.target.value)}
                                type="text"
                                placeholder="Type your tech support question..."
                                className="flex-1 bg-[#2a3343] text-gray-100 text-sm rounded-md px-4 py-2.5
                                         border-0 focus:outline-none focus:ring-1 focus:ring-blue-500/50
                                         placeholder-gray-500"
                                disabled={isTyping}
                            />
                            <button 
                                type="submit"
                                className="bg-blue-500 text-white p-2.5 rounded-md
                                         hover:bg-blue-600 focus:outline-none
                                         disabled:opacity-50 min-w-[40px]
                                         transition-colors flex items-center justify-center"
                                disabled={isTyping || !newMessage.trim()}
                                aria-label="Send message"
                            >
                                <FontAwesomeIcon icon={faPaperPlane} className="text-sm" />
                            </button>
                        </form>
                    </div>
                </div>
            </div>
        </>
    );
} 