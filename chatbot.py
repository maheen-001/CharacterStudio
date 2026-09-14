# This file was made by Maheen Abbasi on Sep. 11, 2026 as part of a personal project

# Imports
import ollama

class CharacterChatbot:
    """
    A simple chatbot that uses a local Ollama-hosted LLM to roleplay as a given character.
    Based on a name/personality desc from the user along with te running conversation history.
    """

    # Set default to the lightweight model (may change later, I want faster response times)
    def __init__(self, model_name = "llama3.2:1b"):
        self.model_name = model_name

    def respond(self, message: str, chat_history: list, char_name: str, char_personality: str, user_alias: str = None):
        """
        respond: generate the character's next reply to a user message.

        Input(s):
            message: the latest user message to respond to.
            chat_history: A list of dicts to represent prior turns in the convo, eahc with "role (user/assistant) and "content" keys (with oldest first).
            char_name: The name of the character bein roleplayed; injected here to maintain the persona
            char_personality: a short dec of the character's traits + personality; again injected here to maitain persona and to steer the tone + behavior
            user_alias: the user's chosen alias from user_alias_state
        
        Output(s): returns the character's reply as a string, extracted from the Ollama chat response.
            --> Hard-coded cap of 4 sentences and order to stay in character at all costs. Can change to make it more details but I need it to be fast rn.
        """

        # Build a system message injecting the persona's rules
        system_prompt = (
            f"You are roleplaying strictly as {char_name}. "
            f"Your traits and personality: {char_personality}. "
            f"Stay in character at all costs. Keep answers under 3 sentences."
        )
        if user_alias:
            system_prompt += f" You are speaking with someone who goes by {user_alias}."

        # Assemble the prompt payload (system prompt -> full history -> new user message)
        messages = [{"role": "system", "content": system_prompt}]

        # Normalize each history entry's content since Gradio can return it as EITHER a plain string or a list of content-part dicts
        for entry in chat_history:
            messages.append({
                "role": entry["role"],
                "content": _flatten_content(entry["content"]),
            })

        messages.append({"role": "user", "content": message})

        # Query the local Ollama API and return just the reply text
        response = ollama.chat(model = self.model_name, messages = messages)
        return response['message']['content']

def _flatten_content(content):
    """
    flatten_content: a helper method that normalizes a Gradio chat message's content field into a plain string
    """

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text")
    return str(content)