# This file was made by Maheen Abbasi on Sep. 11, 2026 as part of a personal project

"""
app.py: A self-contained Gradio web app that walks a user through a 4-step flow:
    1. enter a display alias
    2. build a character via a styled form
    3. optionally generate an avatar 
    4. chat with the character via a locally-hosted LLM (Ollama), with the avatar generated through Stable Diffusion.

    - A siidebar lists up to MAX_CHARACTERS previously created characters with a button to start building a new one.
    - The bot sends the first message (as an in-character greeting) rather than waiting on the user, and replies stream in by tokens
    with a bouncing dots indicater while Ollama is still warming up and/or generating.
    - Screens are implemented as toggled gr.Column's within a single Blocks layout, so only one is visible at a time, driven by button click callbacks.
    - The sidebar is implemented as MAX_CHARACTERS pre-built rows, toggled visible/hidden as characters are added.

    Run directly (python app.py on Windows) to launch the Gradio interface in a browser.
        -> Shoudl run locally as long as necessary GPU/CPU resources exist and Ollama is installed with the target model (llama3:8b) already pulled.

"""

# Imports
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
from image_gen import AvatarGenerator
from chatbot import CharacterChatbot

# Initialize models
avatar_gen = AvatarGenerator()
bot = CharacterChatbot(model_name = "llama3:8b")

# Arbitrary, but I am gonna make max number of saved characters that the sidebar holds = 10
#    -> Get rid of the oldest one with FIFO
MAX_CHARACTERS = 10

TYPING_INDICATOR_HTML = (
    '<span class="typing-indicator">'
    '<span class="dot"></span><span class="dot"></span><span class="dot"></span>'
    "</span>"
)

def make_placeholder_avatar(name: str) -> Image.Image:
    """
    make_placeholder_avatar: Builds a simple solid-color square avatar bearing the character's initial, used 
    when the user skips Stable Diffusion generation, so every character still has an image to show in the sidebar gallery.

    Input(s):
        name: the character's name; only the first character is used.

    Output(s): a PIL.Image.Image, 512x512, matching the app's pink accent.
    """
    img = Image.new("RGB", (512, 512), color="#E8709F")
    draw = ImageDraw.Draw(img)
    initial = name.strip()[0].upper() if name.strip() else "?"
    try:
        font = ImageFont.truetype("arial.ttf", 220)
    except Exception:
        # Falls back to PIL's built-in bitmap font if arial.ttf isn't available on this system; smaller, but still a usable placeholder.
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), initial, font = font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(
        ((512 - w) / 2 - bbox[0], (512 - h) / 2 - bbox[1]),
        initial,
        fill="#FFFFFF",
        font=font,
    )
    return img

"""
STYLING
"""

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500&display=swap');

.gradio-container {
    --bg: #FFFFFF;
    --panel: #FFF3F7;
    --panel-border: #F3C6D9;
    --accent: #E23F84;
    --accent-soft: rgba(226, 63, 132, 0.12);
    --text: #2B1420;
    --text-muted: #A9788C;
    --error: #D64550;

    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'IBM Plex Sans', sans-serif;
}

.gradio-container h1, .gradio-container h2, .gradio-container h3 {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    letter-spacing: -0.01em;
    color: var(--text) !important;
}

.screen {
    max-width: 640px;
    margin: 0 auto;
    padding: 3rem 1.5rem;
}
.step-caption {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.8rem;
    letter-spacing: 0.03em;
    color: var(--text-muted) !important;
    margin-bottom: 0.5rem;
}
.welcome-heading h2 { font-size: 2rem; line-height: 1.25; margin-bottom: 1.5rem; }
.alias-input textarea, .alias-input input {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid var(--panel-border) !important;
    border-radius: 0 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.5rem !important;
    color: var(--accent) !important;
    padding: 0.5rem 0 !important;
    box-shadow: none !important;
}
.alias-input textarea:focus, .alias-input input:focus { border-bottom-color: var(--accent) !important; }

.form-card {
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 12px;
    padding: 1.75rem;
    position: relative;
}
.form-card label { font-family: 'IBM Plex Sans', sans-serif !important; font-size: 0.85rem !important; color: var(--text-muted) !important; }
.form-card textarea, .form-card input { background: var(--bg) !important; border: 1px solid var(--panel-border) !important; color: var(--text) !important; }

.popup-card {
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 12px;
    padding: 2rem;
    max-width: 420px;
    margin: 3rem auto;
    text-align: center;
}

.popup-loading {
    margin: 1.5rem 0 !important;
}

.btn-primary, .btn-primary button {
    background: var(--accent) !important;
    color: #FFFFFF !important;
    border: none !important;
    font-weight: 600 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
.btn-secondary, .btn-secondary button {
    background: transparent !important;
    color: var(--text-muted) !important;
    border: 1px solid var(--panel-border) !important;
}
.field-error { color: var(--error) !important; font-size: 0.9rem; }

/* Sidebar: fills the full side, not a boxed-in gallery */
.sidebar {
    display: flex;
    flex-direction: column;
    height: 100%;
    min-height: 100vh;
    padding: 0;
    background: var(--panel);
    border-right: 1px solid var(--panel-border);
}
.sidebar-new-btn, .sidebar-new-btn button { margin: 1rem 1rem 0.5rem 1rem; width: calc(100% - 2rem); }
.sidebar-row {
    align-items: center;
    gap: 0.75rem;
    padding: 0.5rem 1rem;
}

.sidebar-avatar {
    width: 40px !important;
    height: 40px !important;
}
.sidebar-avatar > div {
    width: 40px !important;
    height: 40px !important;
}
.sidebar-avatar img {
    width: 40px !important;
    height: 40px !important;
    object-fit: cover !important;
    object-position: center !important;
    border-radius: 50%;
}

.chat-header-avatar {
    width: 44px !important;
    height: 44px !important;
}
.chat-header-avatar > div {
    width: 44px !important;
    height: 44px !important;
}
.chat-header-avatar img {
    width: 44px !important;
    height: 44px !important;
    object-fit: cover !important;
    object-position: center !important;
    border-radius: 50%;
}

.sidebar-avatar .icon-buttons,
.sidebar-avatar button[aria-label="Fullscreen"],
.sidebar-avatar button[aria-label="Download"],
.chat-header-avatar .icon-buttons,
.chat-header-avatar button[aria-label="Fullscreen"],
.chat-header-avatar button[aria-label="Download"] {
    display: none !important;
}

.sidebar-name-btn, .sidebar-name-btn button {
    background: none !important;
    border: none !important;
    box-shadow: none !important;
    color: var(--text) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 0 !important;
}

/* Chat screen: WhatsApp-style header (small circular avatar + name) with
   the chat itself taking up the bulk of the screen */
.chat-header {
    align-items: center;
    gap: 0.75rem;
    padding: 0.75rem 1rem;
    border-bottom: 1px solid var(--panel-border);
}

.chat-header-name { font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 1.15rem; }

/* Bouncing typing-dots indicator, shown as a chat bubble's content while
   waiting on Ollama */
.typing-indicator { display: inline-flex; gap: 4px; align-items: center; padding: 2px 0; }
.typing-indicator .dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--text-muted);
    animation: bounce 1.2s infinite ease-in-out;
}
.typing-indicator .dot:nth-child(2) { animation-delay: 0.15s; }
.typing-indicator .dot:nth-child(3) { animation-delay: 0.3s; }
@keyframes bounce {
    0%, 80%, 100% { transform: translateY(0); opacity: .4; }
    40% { transform: translateY(-5px); opacity: 1; }
}
"""

with gr.Blocks(title = "Character Studio") as demo:

    # Hold the alias the user picked in screen 1. Threaded through to the chat screen for a personalized greeting and system prompt context.
    user_alias_state = gr.State("")

    # List of saved character dicts: {"name", "personality", "appearance", "avatar" (PIL.Image), "history" (list of role/content dicts)}
    characters_state = gr.State([])

    # Index into characters_state for whichever character is currently active in the chat screen (None when building a new one)
    current_index = gr.State(None)

    # SCREEN 1: alias entry
    with gr.Column(visible = True, elem_classes = "screen") as screen_welcome:
        gr.Markdown("Step 1 of 4", elem_classes = "step-caption")
        gr.Markdown(
            "## What should we call you here?\nPick any name, it's just for this chat!",
            elem_classes = "welcome-heading",
        )
        alias_input = gr.Textbox(
            placeholder = "Type a name...", show_label = False, elem_classes = "alias-input"
        )
        alias_error = gr.Markdown(visible = False, elem_classes = "field-error")
        continue_to_form_btn = gr.Button("Continue", elem_classes = "btn-primary")

    # MAIN APP: sidebar + the character creation / avatar / chat screens. Hidden until the alias is submitted.
    with gr.Row(visible = False) as app_row:

        # Sidebar: + New Character button, then MAX_CHARACTERS pre-built rows (avatar | name), each hidden until a character occupies that slot.
        with gr.Column(scale = 1, min_width = 240, elem_classes = "sidebar"):
            new_char_btn = gr.Button("+ New Character", elem_classes = "btn-primary sidebar-new-btn")

            sidebar_rows = []
            for _slot in range(MAX_CHARACTERS):
                with gr.Row(visible = False, elem_classes = "sidebar-row") as _row:
                    with gr.Column(scale = 0, min_width = 40):
                        _avatar = gr.Image(
                            show_label = False, container = False, interactive = False,
                            elem_classes = "sidebar-avatar",
                        )
                    with gr.Column(scale = 1):
                        _name_btn = gr.Button("", elem_classes = "sidebar-name-btn")
                sidebar_rows.append({"row": _row, "avatar": _avatar, "name_btn": _name_btn})

            # Flattened list of every sidebar component in a fixed order, used whenever a callback needs to refresh the whole sidebar at once.
            sidebar_output_components = []
            for _slot in sidebar_rows:
                sidebar_output_components += [_slot["row"], _slot["avatar"], _slot["name_btn"]]

        # Main content
        with gr.Column(scale = 3):   

            # SCREEN 2: character creation form
            with gr.Column(visible = True, elem_classes = "screen") as screen_form:
                gr.Markdown("Step 2 of 4", elem_classes = "step-caption")
                gr.Markdown("## Build your character")
                with gr.Group(elem_classes = "form-card"):
                    char_name = gr.Textbox(label = "Name", value = "Baymax")
                    char_personality = gr.Textbox(
                        label = "Personality",
                        value = "Baymax is a gentle, caring healthcare robot who prioritizes helping and protecting others. He is extremely patient, polite, and innocent, often taking things literally and misunderstanding jokes or sarcasm. He speaks calmly and simply, with a soft, reassuring demeanor. Despite being a robot, he is deeply compassionate and loyal to those he cares about.",
                        lines = 3,
                    )
                    char_appearance = gr.Textbox(
                        label = "Appearance",
                        value = "Baymax is a large, inflatable white healthcare robot with a round, soft-looking body, short arms and legs, and simple black facial features: two dots connected by a thin line. His design is smooth, minimal, and friendly, giving him a cuddly, approachable appearance.",
                        lines = 2
                    )
                form_error = gr.Markdown(visible = False, elem_classes = "field-error")
                continue_to_avatar_btn = gr.Button("Continue", elem_classes = "btn-primary")

            # SCREEN 3: avatar decision "popup" for generating the image
            with gr.Column(visible = False, elem_classes = "screen") as screen_popup:
                gr.Markdown("Step 3 of 4", elem_classes = "step-caption")
                with gr.Group(elem_classes = "popup-card"):
                    gr.Markdown("### Generate an avatar?")
                    gr.Markdown(
                        "This renders a portrait locally and can take a minute or two"
                        " depending on your hardware."
                    )
                    # Tell user the avatar is being generated
                    popup_loading = gr.Markdown("Generating your character...", visible = False, elem_classes = "step-caption popup-loading")
                    with gr.Row():
                        generate_avatar_btn = gr.Button("Generate avatar", elem_classes = "btn-primary")
                        skip_avatar_btn = gr.Button("Skip for now", elem_classes = "btn-secondary")

            # SCREEN 4: chat
            with gr.Column(visible = False, elem_classes = "screen chat-screen") as screen_chat:
                with gr.Row(elem_classes = "chat-header"):
                    with gr.Column(scale = 0, min_width = 44):
                        avatar_display = gr.Image(
                            show_label = False, container = False, interactive = False,
                            elem_classes = "chat-header-avatar",
                        )
                    with gr.Column(scale = 1):
                        chat_header_name = gr.Markdown(elem_classes = "chat-header-name")

                # sanitize_html = False lets the typing-dots HTML render instead of showing as literal escaped tag text.
                chatbot_ui = gr.Chatbot(show_label = False, elem_classes = "chat-window", sanitize_html = False)
                msg_input = gr.Textbox(placeholder = "Message...", show_label = False)
                clear_btn = gr.ClearButton([msg_input, chatbot_ui])

    # HELPERS

    def build_sidebar_updates(characters):
        """
        build_sidebar_updates: builds the list of Gradio updates for every sidebar slot (up to MAX_CHARACTERS), so the sidebar mirrors
        characters_state. 
            -> Slots beyond len(characters) are hidden.

        Input(s): characters list

        Output(s): a flat list ordered as [row0, avatar0, name_btn0, row1, ...] matching sidebar_output_components, so it can be spread directly into
        a callback's return tuple.
        """

        updates = []
        for i in range(MAX_CHARACTERS):
            if i < len(characters):
                c = characters[i]
                updates += [gr.update(visible = True), gr.update(value = c["avatar"]), gr.update(value = c["name"])]
            else:
                updates += [gr.update(visible = False), gr.update(value = None), gr.update(value = "")]
        return updates

    def no_op_sidebar_updates():
        """
        no_op_sidebar_updates: used during token-by-token streaming so we aren't recomputing all 10 sidebar rows on every single chunk.
            -> leaves everything as is, basically
        """
        return [gr.update() for _ in range(MAX_CHARACTERS * 3)]

    def load_character_by_index(idx, characters):
        """
        load_character_by_index: shared logic for loading a saved character into the chat screen, used by each sidebar row's click handler.

        Input(s):
            idx: position of click
            characters: list of characters

        Output(s): a tuple of (current_index, avatar_display value, chatbot_ui history, chat_header_name text, char_name, char_personality,
        char_appearance, screen_form visibility, screen_popup visibility, screen_chat visibility).
        """
        c = characters[idx]

        return (
            idx,
            c["avatar"], c["history"], f"**{c['name']}**",
            c["name"], c["personality"], c["appearance"],
            gr.update(visible = False), gr.update(visible = False), gr.update(visible = True),
        )
    
    # CALLBACKS

    def submit_alias(alias):
        """
        submit_alias: a Gradio callback function for continuing past the alias screen.

        Input(s):
            alias: the raw text from alias_input

        Output(s): a tuple of Gradio updates controlling screen_welcome's visibility (stays visible on failure), screen_form's visibility (visible on success)
        user_alias's new value, and alias_error's visibility (shown if the field was left blank).
        """

        # Get the alias
        if not alias or not alias.strip():
            return (
                gr.update(visible = True),
                gr.update(visible = False),
                gr.update(),
                gr.update(value = "Please enter a name to continue.", visible = True),
            )

        # Return the alias
        return (
            gr.update(visible = False),
            gr.update(visible = True),
            alias.strip(),
            gr.update(visible = False),
        )

    continue_to_form_btn.click(
        fn = submit_alias,
        inputs = [alias_input],
        outputs = [screen_welcome, app_row, user_alias_state, alias_error],
    )

    # Allow pressif enter in the alias box as well just for ease
    alias_input.submit(
        fn = submit_alias,
        inputs = [alias_input],
        outputs = [screen_welcome, app_row, user_alias_state, alias_error],
    )

    def submit_character(name, personality, appearance):
        """
        submit_character: a Gradio callback for continuing pst the character form screen.

        Input(s):
            name, personality, appearance: current calues of the three form fieklds
        
        Output(s): a tuple of Gradio updates controlling screen_form's visibility (stays visible if any field is blank), screen_popups
        visibility (visible of success), and form_error's visibility/text.
        """

        if not name.strip() or not personality.strip() or not appearance.strip():
            return (
                gr.update(visible = True),
                gr.update(visible = False),
                gr.update(value = "Fill in all three fields to continue.", visible = True),
            )
        return gr.update(visible = False), gr.update(visible = True), gr.update(visible = False)

    continue_to_avatar_btn.click(
        fn = submit_character,
        inputs = [char_name, char_personality, char_appearance],
        outputs = [screen_form, screen_popup, form_error],
    )

    def _finish_character_creation(new_char, alias, characters):
        """
        _finish_character_creation: shared logic for both do_generate_avatar and skip_avatar. 
        Saves the new character (popping the oldest if at MAX_CHARACTERS), reveals the chat screen, and streams 
        an in-character greeting as the first chat message.

        This is a GENERATOR: yields multiple times so the UI shows a typing indicator immediately, then fills in the greeting as it
        streams in from Ollama.
        """

        characters = list(characters)
        if len(characters) >= MAX_CHARACTERS:
            # evict oldest if needed
            characters.pop(0)
        # Add new character
        characters.append(new_char)
        new_index = len(characters) - 1

        # FIRST YIELD: reveal the chat screen immediately with a typing indicator in place of the greeting, and a full sidebar refresh
        #    -> the new character's row needs to appear
        display_history = [{"role": "assistant", "content": TYPING_INDICATOR_HTML}]
        yield (
            new_char["avatar"], f"**{new_char['name']}**",
            gr.update(visible = False), gr.update(visible = True),
            characters, new_index, display_history,
            *build_sidebar_updates(characters), gr.update(interactive = False),
        )

        # Stream the greeting in, overwriting the typing indicator with growing text as tokens arrive
        #     -> Sidebar/avatar/header stay unchanged during this loop (no_op_sidebar_updates)
        for partial_greeting in bot.greet(new_char["name"], new_char["personality"], user_alias = alias):
            display_history[-1]["content"] = partial_greeting
            yield (
                gr.update(), gr.update(),
                gr.update(), gr.update(),
                characters, new_index, display_history,
                *no_op_sidebar_updates(), gr.update(),
            )

        # Push the completed greeting into character's saved history and re-enable input
        new_char = dict(new_char)
        new_char["history"] = display_history
        characters[new_index] = new_char
        yield (
            gr.update(), gr.update(),
            gr.update(), gr.update(),
            characters, new_index, display_history,
            *no_op_sidebar_updates(), gr.update(interactive = True),
        )

    def do_generate_avatar(appearance, alias, name, personality, characters):
        """
        do_generate_avatar: Gradio callback for the generate avatar button on the popup screen. Disables bth popup buttons,
        shows a loading message, then runs Stable Diffusion inference before handing off to _finish_character_creation for saving + the streamed greeting.

        Input(s):
            appearance: value of char_appearance, used as the image gen prompt
            alias: the user's chosen alias
            name, personality: of the character
            characters: list of existing characters

        Output(s): a tuple of the generated PIL.Image.Image for avatar_display, screen popup's visibility (hidden), screen_chat's visibility (shown), and the greeting Markdown text.
        """

        # FIRST YIELD: locks the popup before starting the SD call (everyting unrelated to the lock is just gr.update() since ntohing else should change yet)
        yield (
            gr.update(), gr.update(), gr.update(), gr.update(),
            characters, gr.update(), gr.update(),
            *no_op_sidebar_updates(), gr.update(),

            # Lock generate and skip avatar buttons
            gr.update(interactive = False), gr.update(interactive = False),

            # Show loading popup
            gr.update(visible = True),
        )
        img = avatar_gen.generate_avatar(appearance)
        avatar_gen.release_gpu_memory()

        new_char = {"name": name, "personality": personality, "appearance": appearance, "avatar": img, "history": []}

        for step in _finish_character_creation(new_char, alias, characters):
            yield (*step, gr.update(interactive = True), gr.update(interactive = True), gr.update(visible = False))

    generate_avatar_btn.click(
        fn = do_generate_avatar,
        inputs = [char_appearance, user_alias_state, char_name, char_personality, characters_state],
        outputs = [avatar_display, chat_header_name, screen_popup, screen_chat, characters_state, current_index, chatbot_ui, *sidebar_output_components, msg_input, generate_avatar_btn, skip_avatar_btn, popup_loading],
    )

    def skip_avatar(alias, name, personality, appearance, characters):
        """
        skip_avatar: a Gradio callback for the skip for now button on the popup screen (for the avatar gen). Same as do+generate_avatar,
        but uses a generated placeholder image instead of running Stable Diffusion.

        Input(s):
            alias: the uswr's chosen alias from user_alias_state
            name, personality, appearance: of the created character
            characters: list of existing characters
        
            Output(s): a tuple of screen_popup's visibility (hidden), screen_chat's visibility(shown), and the greeting Markdown text
        """

        yield (
            gr.update(), gr.update(), gr.update(), gr.update(),
            characters, gr.update(), gr.update(),
            *no_op_sidebar_updates(), gr.update(),
        
            # Lock generate and skip avatar buttons
            gr.update(interactive = False), gr.update(interactive = False),
        
            # Show loading popup
            gr.update(visible = True),
        )
        
        placeholder = make_placeholder_avatar(name)
        new_char = {"name": name, "personality": personality, "appearance": appearance, "avatar": placeholder, "history": []}

        for step in _finish_character_creation(new_char, alias, characters):
            yield (*step, gr.update(interactive = True), gr.update(interactive = True), gr.update(visible = False))


    skip_avatar_btn.click(
        fn = skip_avatar,
        inputs = [user_alias_state, char_name, char_personality, char_appearance, characters_state],
        outputs = [avatar_display, chat_header_name, screen_popup, screen_chat, characters_state, current_index, chatbot_ui, *sidebar_output_components, msg_input, generate_avatar_btn, skip_avatar_btn, popup_loading],
    )

    def start_new_character():
        """
        start_new_character: Gradio callback for the sidebar's + New Character button. Clears the form fields, unselects
        any active characters, and jumps back to screen 2 (Character Creation)
        """
        return (
            # blank character name, personality, and appearance
            "", "", "",

            # current index reset
            None,

            # Hide chat screen, popups, and error forms; show the character creation form
            gr.update(visible = False), gr.update(visible = False), gr.update(visible = True), gr.update(visible = False),

            # Reser avatar popup buttons and hides loading popup
            gr.update(interactive = True), gr.update(interactive = True), gr.update(visible = False)
        )

    new_char_btn.click (
        fn = start_new_character,
        inputs = [],
        outputs = [char_name, char_personality, char_appearance, current_index, screen_chat, screen_popup, screen_form, form_error, generate_avatar_btn, skip_avatar_btn, popup_loading],
    )

    # Wire each of the MAX_CHARACTERS sidebar rows to load that character on click.
    for i, slot in enumerate(sidebar_rows):
        slot["name_btn"].click(
            # slot_index = i binds each button to its own fixed index at definition time
            fn = lambda characters, slot_index = i: load_character_by_index(slot_index, characters),
            inputs = [characters_state],
            outputs = [
                current_index, avatar_display, chatbot_ui, chat_header_name,
                char_name, char_personality, char_appearance,
                screen_form, screen_popup, screen_chat,
            ],
        )

    def user_chat(user_msg, alias, characters, idx):
        """
        user_chat: Gradio callback for when the user submits a chat message. It:
        
            1. Sends the message along with running history, current character settings, and the user's alias to the chatbot
            2. Appends the exchange to the visible chat history
            3. Clears the input box
        msg_input (ability to send messages) is blocked suring generation to prevent sedning a seconf message mid-reply.
        
            Input(s):
                user_msg: The text that the user just typed and submitted
                alias: the user's chosen alias from user_alias_state
                characters: the list of characters
                idx: the active character's index from current_index
        
        Generator function, uses yield instead of return to re-render the chat window multiple times during a single submission
        rather than only once at the end so thaty there's a streaming/typing effect (instead of one big message all of a sudden).
        """

        # No character selected (idx = None) or index is out of range, clear the input box, leave chat as-is, and exit early
        if idx is None or idx >= len(characters):
            yield "", [], characters
            return

        # Get the full dict of active character and their history
        characters = list(characters)
        char = characters[idx]
        prior_history = char["history"]

        # Show the user's message immediately, with an empty assistant bubble that fills in as tokens stream in
        display_history = prior_history + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": ""},
        ]

        # FIRST YIELD: pushes the user's message (and the empty assistant bubble) to the UI; disables input
        yield gr.update(value = "", interactive = False), display_history, characters

        # bot.respond is the generator from chatbot.py; each yield from it hands back the reply accumulated so far
        for partial_reply in bot.respond(
            user_msg, prior_history, char["name"], char["personality"], user_alias=alias
        ):
            # Overwrite the last item in display_history (the empty or partial assistant bubble) with the latest text
            display_history[-1]["content"] = partial_reply
            # SECOND YIELD: re-render the chat window, acrtually update with new text
            yield gr.update(), display_history, characters

        # Ollama message completed, update character's record to preserve history
        updated_char = dict(char)
        updated_char["history"] = display_history
        characters[idx] = updated_char

        # THIRD YIELD: pushes the saved state to the UI one more time, guarsanteeing that the UI reflects the fully saved characters_state, not jsut the local display_history
        #    -> Also re-enable input
        yield gr.update(interactive = True), display_history, characters

    msg_input.submit (
        fn = user_chat, 
        inputs = [msg_input, user_alias_state, characters_state, current_index], 
        outputs = [msg_input, chatbot_ui, characters_state]
    )

if __name__ == "__main__":
    demo.launch(css = CUSTOM_CSS)