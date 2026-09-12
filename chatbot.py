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

    def respond(self, message: str, chat_history: list, char_name: str, char_personality: str):
        """
        respond: generate the character's next reply to a user message.

        Input(s):
            message: the latest user message to respond to.
            chat_history: A list of tuples (user_msg, bot_msg) representing the prior turns in the convo, with oldest first.
                --> Each pair is converted into a "user" and then "assistant" message so the model has conversational context.
            char_name: The name of the character bein roleplayed; injected here to maintain the persona
            char_personality: a short dec of the character's traits + personality; again injected here to maitain persona and to steer the tone + behavior
        
        Output(s): returns the character's reply as a string, extracted from the Ollama chat response.
            --> Hard-coded cap of 4 sentences and order to stay in character at all costs. Can change to make it more details but I need it to be fast rn.
        """

        # Build a system message injecting the persona's rules
        system_prompt = (
            f"You are roleplaying strictly as {char_name}. "
            f"Your traits and personality: {char_personality}. "
            f"Stay in character at all costs. Keep answers under 3 sentences."
        )

        # Assemble the prompt payload (system prompt -> full history -> new user message)
        messages = [{"role": "system", "content": system_prompt}]
        for user_msg, bot_msg in chat_history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": bot_msg})

        messages.append({"role": "user", "content": message})

        # Query the local Ollama API and return just the reply text
        response = ollama.chat(model = self.model_name, messages = messages)
        return response['message']['content']