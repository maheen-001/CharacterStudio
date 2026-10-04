# This file was made by Maheen Abbasi on Sep. 11, 2026 as part of a personal project

# Imports
import ollama

# Send onlt this many of the most recent messages to Ollama each turn; anything older is expected to
# already be folded into the character's memory_summary (by the caller).
MAX_RECENT_MESSAGES = 16

class CharacterChatbot:
    """
    A simple chatbot that uses a local Ollama-hosted LLM to roleplay as a given character.
    Based on a name/personality desc from the user along with te running conversation history.
    """

    # Set default to the lightweight model (may change later, I want faster response times)
    def __init__(self, model_name = "llama3:8b"):
        self.model_name = model_name


    def _build_system_prompt(self, char_name, char_personality, char_lorebook = None, user_profile = None, memory_summary = None):
        """
        _build_system_prompt: shared prompt-building logic used by both respond() and greet(), so the two
        stay consistent with each other as more persona/profile fields get added later instead of drifting apart.

        Input(s):
            char_name, char_personality: same as respond()/greet()
            char_lorebook: optional worldbuilding notes (setting, backstory, other characters, rules) the
                character should stay consistent with
            user_profile: optional dict of the user's chosen alias/gender/age/context, from user_profile_state.
                Any key can be missing/empty; only what's present gets added to the prompt.
            memory_summary: a summary of the last 16 messages to improve the bot's memory

        Output(s): the assembled system prompt string.
        """

        # Build a system message injecting the persona's rules
        system_prompt = (
            f"You are roleplaying strictly as {char_name} in an immersive, narrative chat, in the "
            f"style of Character.AI. Your traits and personality: {char_personality}. "
            f"Never narrate or speak for the user. Stay in character at all costs.\n\n"
            f"FORMATTING -- follow this structure exactly:\n"
            f"- Alternate between short action/thought beats and spoken dialogue, each on its OWN line, "
            f"separated by a blank line. Never combine an action and a line of dialogue into the same line.\n"
            f'- Wrap every action, gesture, or internal thought in *asterisks*, e.g. *tilts head, considering this*\n'
            f'- Wrap every spoken line in "quotation marks", e.g. "That\'s an interesting question."\n'
            f"- Use 2-4 beats/lines total per reply (mix of action and dialogue), more only for emotionally "
            f"significant moments. Each individual line should be a sentence or two -- not a full paragraph "
            f"crammed together.\n\n"
            f"Example of the exact shape to follow (content is just an illustration, not your actual personality):\n"
            f"*She glances up from the book, a slow smile spreading across her face.*\n\n"
            f'"I wasn\'t expecting you to come find me here."\n\n'
            f"*She sets the book down, patting the seat beside her.*\n\n"
            f'"Sit with me for a while?"'
        )

        # Fold in the lorebook (world notes) if one was provided
        if char_lorebook:
            system_prompt += f" World/background notes to stay consistent with: {char_lorebook}"

        # Give a memory summary for the bot to use (to improve memory and responses)
        if memory_summary:
            system_prompt += (
                f" Summary of earlier parts of this conversation, for your own reference (don't repeat "
                f"it verbatim, just stay consistent with it): {memory_summary}"
            )

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


    def respond(self, message: str, chat_history: list, char_name: str, char_personality: str, char_lorebook: str = None, user_profile: dict = None, memory_summary = None):
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
            memory_summary: a summary of the last 16 messages to improve the bot's memory
        
        Yields: the character's reply text so far as a string, growing with each new chunk recieved frm Ollama. The final yielded value is the compelte reply.
            --> Hard-coded cap of 3 sentences and order to stay in character at all costs. Can change to make it more details but I need it to be fast rn.
        """

        # Build the system prompt (persona rules + lorebook + user profile, via the shared helper)
        system_prompt = self._build_system_prompt(char_name, char_personality, char_lorebook, user_profile, memory_summary)

        # Assemble the prompt payload (system prompt -> full history -> new user message)
        messages = [{"role": "system", "content": system_prompt}]

        # Only the most recent window of turns foes in verbatim, older context still lives in memory_summary instead
        recent_history = chat_history[-MAX_RECENT_MESSAGES:]
        for entry in recent_history:
            messages.append({"role": entry["role"], "content": _flatten_content(entry["content"])})

        messages.append({"role": "user", "content": message})

        # Query the local Ollama API in STREAMING mode: returns an iterator of small response chunks instead of one full response
        #    -> accumulate and yield the growing reply as each chunk arrives
        partial_reply = ""
        stream = ollama.chat(model = self.model_name, messages = messages, stream = True)
        for chunk in stream:
            token = chunk.get("message", {}).get("content", "")
            partial_reply += token
            yield partial_reply


    def greet(self, char_name: str, char_personality: str, char_lorebook: str = None, user_profile: dict = None, memory_summary = None):
        """
        greet(): generates an in-character opening message from the bot, starting a conversation before the user has saif anything.
        This uses the same streaming-generator design as respond(), but with no chat_history and no user message. Model is prompted
        only by the system instruction to make an opening line.
            -> helps Ollama warm up and bypasses awkward waiting time for the first message.

        Input(s):
            char_name, char_personality: same as respond()
            char_lorebook: also same as respond()
            user_profile: also same as respond()
            memory_summary: a summary of the last 16 messages to improve the bot's memory

        Yields: the greeting text so far, growing with each chunk received from Ollama.
        """

        # Same setup and prompting as in respond(), via the shared helper
        system_prompt = self._build_system_prompt(char_name, char_personality, char_lorebook, user_profile, memory_summary)
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


    def summarize(self, turns_to_fold, char_name, existing_summary = None):
        """
        summarize: a non-streaming helper that condenses a chunk of older chat turns into a short
        running summary so that prompts sent to Ollama stay a manageable size no matter how long the 
        conversation gets.

        Input(s):
            turns_to_fold: slice of chat history (oldest first) that needs to be folded in
            char_name: same as respond()
            existing_summary: the chara's current memory_summary (if any). New context gets merged onto it.
        
        Outputs/Returns: the new, updated summary string, which should be about 3 to 5 sentences.
        """
        # Get the messages to summarize
        convo_text = "\n".join(
            f"{'User' if entry['role'] == 'user' else char_name}: {_flatten_content(entry['content'])}"
            for entry in turns_to_fold
        )

        # Send a prompt to summarize those messages
        prompt = (
            f"Summarize the key facts, events, and emotional beats from this part of a roleplay "
            f"conversation in 3-5 concise sentences, third person, so {char_name} can remember them "
            f"later. Prioritize concrete details (names, decisions, promises, relationship changes) "
            f"over flavor text."
        )

        # Build on existing memory_summary if it exists
        if existing_summary:
            prompt += f" Build on and merge with this earlier summary rather than replacing it: {existing_summary}"
        prompt += f"\n\nConversation:\n{convo_text}"

        # Get and return
        response = ollama.chat(model = self.model_name, messages = [{"role": "user", "content": prompt}], stream = False)
        return response.get("message", {}).get("content", "").strip()


def _flatten_content(content):
    """
    flatten_content: a helper method that normalizes a Gradio chat message's content field into a plain string
    """

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text")
    return str(content)