# chatbot.py - Advanced Gemini-powered chatbot with live updates
import os
import json
import re
from typing import Dict, List, Optional, Tuple
import google.generativeai as genai
from utils_data import load_df
from utils_json import to_builtin
from percentile_modes import summarize_percentiles
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RiskOpsChatbot:
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.model_name = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
        self.conversation_history = []
        
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
        else:
            self.model = None
    
    def get_live_dashboard_state(self) -> Dict:
        """Get current dashboard state for context-aware responses"""
        try:
            df = load_df("data.json")
            if df.empty:
                return {"error": "No data loaded"}
            
            # Get current filters and mode from typical dashboard state
            current_state = {
                "total_submissions": len(df),
                "filtered_count": len(df),  # Would be updated by actual filters
                "current_mode": "balanced",  # Default, could be passed from frontend
                "top_states": df["primary_risk_state"].value_counts().head(5).to_dict(),
                "status_distribution": {
                    "TARGET": len(df[df.get("appetite_status", "") == "TARGET"]) if "appetite_status" in df.columns else 0,
                    "IN": len(df[df.get("appetite_status", "") == "IN"]) if "appetite_status" in df.columns else 0,
                    "OUT": len(df[df.get("appetite_status", "") == "OUT"]) if "appetite_status" in df.columns else 0,
                },
                "avg_metrics": {
                    "premium": float(df["total_premium"].mean()) if "total_premium" in df.columns else 0,
                    "tiv": float(df["tiv"].mean()) if "tiv" in df.columns else 0,
                    "winnability": float(df["winnability"].mean()) if "winnability" in df.columns else 0,
                },
                "recent_activity": {
                    "new_submissions_today": len(df[df["fresh_days"] <= 1]) if "fresh_days" in df.columns else 0,
                    "high_value_submissions": len(df[df["total_premium"] > df["total_premium"].quantile(0.9)]) if "total_premium" in df.columns else 0,
                }
            }
            return current_state
        except Exception as e:
            return {"error": str(e)}
    
    def get_system_context(self) -> str:
        """Get current system context for the chatbot"""
        try:
            df = load_df("data.json")
            if df.empty:
                return "No data currently loaded."
            
            summary = {
                "total_submissions": len(df),
                "states": df["primary_risk_state"].value_counts().head(5).to_dict(),
                "avg_premium": float(df["total_premium"].mean()),
                "avg_tiv": float(df["tiv"].mean()),
                "lobs": df["line_of_business"].value_counts().head(3).to_dict(),
            }
            
            percentiles = summarize_percentiles(df)
            
            context = f"""
RiskOps Underwriting Dashboard Context:
- Total submissions: {summary['total_submissions']}
- Top states: {', '.join([f"{k}({v})" for k, v in list(summary['states'].items())[:3]])}
- Average premium: ${summary['avg_premium']:,.0f}
- Average TIV: ${summary['avg_tiv']:,.0f}
- Main LOBs: {', '.join(summary['lobs'].keys())}

Available Actions:
1. Filter submissions (by state, status, premium range, etc.)
2. Switch modes (unicorn, balanced, loose, turnaround, custom)
3. Search submissions by account name
4. Sort submissions by various criteria
5. Get detailed explanations for specific submissions
6. Analyze portfolio insights and trends

Current percentile ranges:
- Premium: P50=${percentiles['premium'][50]:,.0f}, P90=${percentiles['premium'][90]:,.0f}
- TIV: P50=${percentiles['tiv'][50]:,.0f}, P90=${percentiles['tiv'][90]:,.0f}
- Loss Ratio: P50={percentiles['lr'][50]:.2f}, P90={percentiles['lr'][90]:.2f}
"""
            return context
        except Exception as e:
            return f"Error getting context: {str(e)}"
    
    def parse_action_intent(self, message: str) -> Optional[Dict]:
        """Parse user message for actionable intents"""
        message_lower = message.lower()
        
        # Filter intents
        if any(word in message_lower for word in ['filter', 'show', 'find', 'display']):
            action = {"type": "filter", "params": {}}
            
            # Extract state
            state_match = re.search(r'\b([A-Z]{2})\b', message.upper())
            if state_match:
                action["params"]["state"] = state_match.group(1)
            
            # Extract status
            if 'target' in message_lower:
                action["params"]["status"] = "TARGET"
            elif 'out' in message_lower and 'appetite' in message_lower:
                action["params"]["status"] = "OUT"
            elif 'in' in message_lower and 'appetite' in message_lower:
                action["params"]["status"] = "IN"
            
            # Extract premium range
            premium_matches = re.findall(r'\$?(\d+(?:,\d{3})*(?:\.\d+)?)\s*k?', message)
            if premium_matches:
                amounts = []
                for match in premium_matches:
                    amount = float(match.replace(',', ''))
                    if 'k' in message_lower:
                        amount *= 1000
                    amounts.append(amount)
                
                if len(amounts) >= 2:
                    action["params"]["min_premium"] = min(amounts)
                    action["params"]["max_premium"] = max(amounts)
                elif len(amounts) == 1:
                    if 'over' in message_lower or 'above' in message_lower or '>' in message:
                        action["params"]["min_premium"] = amounts[0]
                    elif 'under' in message_lower or 'below' in message_lower or '<' in message:
                        action["params"]["max_premium"] = amounts[0]
            
            return action
        
        # Mode switching intents
        if any(word in message_lower for word in ['switch', 'change', 'mode']):
            if 'unicorn' in message_lower:
                return {"type": "mode", "params": {"mode": "unicorn"}}
            elif 'balanced' in message_lower:
                return {"type": "mode", "params": {"mode": "balanced"}}
            elif 'loose' in message_lower:
                return {"type": "mode", "params": {"mode": "loose"}}
            elif 'turnaround' in message_lower:
                return {"type": "mode", "params": {"mode": "turnaround"}}
            elif 'custom' in message_lower:
                return {"type": "mode", "params": {"mode": "custom"}}
        
        # Search intents
        if any(word in message_lower for word in ['search', 'look for', 'account']):
            # Extract potential account name
            search_match = re.search(r'(?:for|account)\s+["\']?([^"\']+)["\']?', message, re.IGNORECASE)
            if search_match:
                return {"type": "search", "params": {"q": search_match.group(1).strip()}}
        
        # Sort intents
        if 'sort' in message_lower or 'order' in message_lower:
            sort_params = {"sort_by": "priority_score", "sort_dir": "desc"}
            
            if 'premium' in message_lower:
                sort_params["sort_by"] = "total_premium"
            elif 'tiv' in message_lower:
                sort_params["sort_by"] = "tiv"
            elif 'winnability' in message_lower:
                sort_params["sort_by"] = "winnability"
            elif 'name' in message_lower or 'account' in message_lower:
                sort_params["sort_by"] = "account_name"
            
            if 'ascending' in message_lower or 'asc' in message_lower or 'low to high' in message_lower:
                sort_params["sort_dir"] = "asc"
            
            return {"type": "sort", "params": sort_params}
        
        return None
    
    def generate_confirmation_message(self, action: Dict) -> str:
        """Generate a confirmation message for the action"""
        action_type = action["type"]
        params = action["params"]
        
        if action_type == "filter":
            parts = []
            if "state" in params:
                parts.append(f"state = {params['state']}")
            if "status" in params:
                parts.append(f"status = {params['status']}")
            if "min_premium" in params:
                parts.append(f"minimum premium = ${params['min_premium']:,.0f}")
            if "max_premium" in params:
                parts.append(f"maximum premium = ${params['max_premium']:,.0f}")
            if "q" in params:
                parts.append(f"search term = '{params['q']}'")
            
            return f"Apply filters: {', '.join(parts)}?"
        
        elif action_type == "mode":
            return f"Switch to {params['mode']} mode?"
        
        elif action_type == "search":
            return f"Search for accounts containing '{params['q']}'?"
        
        elif action_type == "sort":
            direction = "ascending" if params["sort_dir"] == "asc" else "descending"
            return f"Sort by {params['sort_by']} in {direction} order?"
        
        return "Execute this action?"
    
    def chat(self, message: str, session_id: str = "default") -> Dict:
        """Main chat method with streaming support"""
        if not self.model:
            return {
                "response": "I'm sorry, but the AI chatbot is not available without a Gemini API key. I can still help with basic navigation using the existing dashboard features.",
                "action": None,
                "streaming": False,
                "ai_used": False
            }
        
        # Add message to conversation history
        self.conversation_history.append({"role": "user", "content": message})
        
        # Get system context
        context = self.get_system_context()
        
        # Get live dashboard state for context
        dashboard_state = self.get_live_dashboard_state()
        
        # Parse for actionable intents
        action = self.parse_action_intent(message)
        
        # Build prompt
        system_prompt = f"""You are an intelligent assistant for the RiskOps Underwriting Dashboard. You help underwriters analyze submissions, navigate the interface, and make data-driven decisions.

{context}

Guidelines:
1. Be helpful, professional, and concise
2. Focus on underwriting and risk assessment topics
3. When users ask for actions (filtering, sorting, mode switching), I'll handle the technical execution
4. Provide insights about the data when relevant
5. Ask clarifying questions when needed
6. Use underwriting terminology appropriately

Current conversation context: The user is interacting with a live dashboard that shows insurance submissions with various filters and modes available.

Live Dashboard State: {dashboard_state}

Remember to reference current data when relevant and suggest specific actions based on what the user is seeing."""
        
        try:
            # Generate response
            full_prompt = f"{system_prompt}\n\nUser: {message}\n\nAssistant:"
            response = self.model.generate_content(full_prompt)
            
            ai_response = response.text.strip() if response.text else "I'm here to help with your underwriting needs."
            
            # Add to conversation history
            self.conversation_history.append({"role": "assistant", "content": ai_response})
            
            # Keep conversation history manageable
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            result = {
                "response": ai_response,
                "action": action,
                "streaming": True,
                "ai_used": True
            }
            
            if action:
                result["confirmation"] = self.generate_confirmation_message(action)
            
            return result
            
        except Exception as e:
            return {
                "response": f"I encountered an error: {str(e)}. Please try rephrasing your question.",
                "action": action,
                "streaming": False,
                "ai_used": False
            }
    
    def get_streaming_response(self, message: str, session_id: str = "default"):
        """Generator for streaming responses"""
        if not self.model:
            yield {
                "type": "error",
                "content": "AI chatbot not available without Gemini API key"
            }
            return
        
        try:
            context = self.get_system_context()
            action = self.parse_action_intent(message)
            
            system_prompt = f"""You are an intelligent assistant for the RiskOps Underwriting Dashboard.

{context}

Be helpful, professional, and provide actionable insights about underwriting and risk assessment."""
            
            full_prompt = f"{system_prompt}\n\nUser: {message}\n\nAssistant:"
            
            # Start streaming response
            yield {"type": "start", "content": ""}
            
            response = self.model.generate_content(full_prompt, stream=True)
            
            accumulated_text = ""
            for chunk in response:
                if chunk.text:
                    accumulated_text += chunk.text
                    yield {
                        "type": "chunk",
                        "content": chunk.text,
                        "accumulated": accumulated_text
                    }
            
            # Send action if detected
            if action:
                yield {
                    "type": "action",
                    "content": action,
                    "confirmation": self.generate_confirmation_message(action)
                }
            
            yield {"type": "end", "content": accumulated_text}
            
        except Exception as e:
            yield {
                "type": "error",
                "content": f"Error: {str(e)}"
            }


# Global chatbot instance
chatbot = RiskOpsChatbot()
