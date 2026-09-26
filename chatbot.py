# This file was made by Maheen Abbasi on Sep. 11, 2026 as part of a personal project

# Imports
import ollama

class CharacterChatbot:
    """
    A simple chatbot that uses a local Ollama-hosted LLM to roleplay as a given character.
    Based on a name/personality desc from the user along with te running conversation history.
    """

    # Set default to the lightweight model (may change later, I want faster response times)
    def __init__(self, model_name = "llama3:8b"):
        self.model_name = model_name

    def _build_system_prompt(self, char_name, char_personality, char_lorebook = None, user_profile = None):
        """
        _build_system_prompt: shared prompt-building logic used by both respond() and greet(), so the two
        stay consistent with each other as more persona/profile fields get added later instead of drifting apart.

        Input(s):
            char_name, char_personality: same as respond()/greet()
            char_lorebook: optional worldbuilding notes (setting, backstory, other characters, rules) the
                character should stay consistent with
            user_profile: optional dict of the user's chosen alias/gender/age/context, from user_profile_state.
                Any key can be missing/empty; only what's present gets added to the prompt.

        Output(s): the assembled system prompt string.
        """

        # Build a system message injecting the persona's rules
        system_prompt = (
            f"You are roleplaying strictly as {char_name}. "
            f"Your traits and personality: {char_personality}. "
            f"Stay in character at all costs. Keep answers under 3 sentences."
        )

        # Fold in the lorebook (world notes) if one was provided
        if char_lorebook:
            system_prompt += f" World/background notes to stay consistent with: {char_lorebook}"

        # Fold in whatever pieces of the user's profile were filled in (all optional)
        if user_profile:
            if user_profile.get("alias"):
                system_prompt += f" You are speaking with someone who goes by {user_profile['alias']}."
            if user_profile.get("gender"):
                system_prompt += f" Their gender: {user_profile['gender']}."
            if user_profile.get("age"):
                system_prompt += f" Their age: {user_profile['age']}."
            if user_profile.get("context"):
                system_prompt += f" Other context about them: {user_profile['context']}"

        return system_prompt

    def respond(self, message: str, chat_history: list, char_name: str, char_personality: str, char_lorebook: str = None, user_profile: dict = None):
        """
        respond: generate the character's next reply to a user message.

        ** This is a GENERATOR function. The caller MUST iterate it, since a single return value can no longer represent a reply that's still being generated.

        Input(s):
            message: the latest user message to respond to.
            chat_history: A list of dicts to represent prior turns in the convo, eahc with "role (user/assistant) and "content" keys (with oldest first).
            char_name: The name of the character bein roleplayed; injected here to maintain the persona
            char_personality: a short dec of the character's traits + personality; again injected here to maitain persona and to steer the tone + behavior
            char_lorebook: optional worldbuilding notes for the character, see _build_system_prompt()
            user_profile: the user's alias/gender/age/context dict from user_profile_state, see _build_system_prompt()
        
        Yields: the character's reply text so far as a string, growing with each new chunk recieved frm Ollama. The final yielded value is the compelte reply.
            --> Hard-coded cap of 3 sentences and order to stay in character at all costs. Can change to make it more details but I need it to be fast rn.
        """

        # Build the system prompt (persona rules + lorebook + user profile, via the shared helper)
        system_prompt = self._build_system_prompt(char_name, char_personality, char_lorebook, user_profile)

        # Assemble the prompt payload (system prompt -> full history -> new user message)
        messages = [{"role": "system", "content": system_prompt}]

        # Normalize each history entry's content since Gradio can return it as EITHER a plain string or a list of content-part dicts
        for entry in chat_history:
            messages.append({
                "role": entry["role"],
                "content": _flatten_content(entry["content"]),
            })

        messages.append({"role": "user", "content": message})

        # Query the local Ollama API in STREAMING mode: returns an iterator of small response chunks instead of one full response
        #    -> accumulate and yield the growing reply as each chunk arrives
        partial_reply = ""
        stream = ollama.chat(model=self.model_name, messages = messages, stream = True)
        for chunk in stream:
            token = chunk.get("message", {}).get("content", "")
            partial_reply += token
            yield partial_reply

    def greet(self, char_name: str, char_personality: str, char_lorebook: str = None, user_profile: dict = None):
        """
        greet(): generates an in-character opening message from the bot, starting a conversation before the user has saif anything.
        This uses the same streaming-generator design as respond(), but with no chat_history and no user message. Model is prompted
        only by the system instruction to make an opening line.
            -> helps Ollama warm up and bypasses awkward waiting time for the first message.

        Input(s):
            char_name, char_personality: same as respond()
            char_lorebook: also same as respond()
            user_profile: also same as respond()

        Yields: the greeting text so far, growing with each chunk received from Ollama.
        """

        # Same setup and prompting as in respond(), via the shared helper
        system_prompt = self._build_system_prompt(char_name, char_personality, char_lorebook, user_profile)
        system_prompt += (
            " This is the very start of the conversation. Greet the user in"
            " character with a short opening line -- don't wait for them to"
            " speak first."
        )

        messages = [{"role": "system", "content": system_prompt}]

        # Same yield
        partial_reply = ""
        stream = ollama.chat(model = self.model_name, messages = messages, stream = True)
        for chunk in stream:
            token = chunk.get("message", {}).get("content", "")
            partial_reply += token
            yield partial_reply

def _flatten_content(content):
    """
    flatten_content: a helper method that normalizes a Gradio chat message's content field into a plain string
    """

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text")
    return str(content)