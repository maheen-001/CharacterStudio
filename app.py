# This file was made by Maheen Abbasi on Sep. 11, 2026 as part of a personal project

"""
app.py: A self contained Gradio web app that lets a user define a custom character (name, personality, and visual appearance),
generate an avatar image for that character through StableDiffusion, and then chat with the character via a locally-hosted LLM (Ollama).

Run directly (`python app.py`) to launch the Gradio interface in a browser.
--> Should run locally as long as necessary GPU/CPU resources exist and Ollama is installed with the target model already pulled.
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

/* Shared screen wrapper: centers content, caps line length */
.screen {
    max-width: 640px;
    margin: 0 auto;
    padding: 3rem 1.5rem;
}

/* Small sequence caption ("Step 1 of 4") */
.step-caption {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.8rem;
    letter-spacing: 0.03em;
    color: var(--text-muted) !important;
    margin-bottom: 0.5rem;
}

/* Welcome screen: the alias input reads like an inline answer, not a
   boxed form field */
.welcome-heading h2 {
    font-size: 2rem;
    line-height: 1.25;
    margin-bottom: 1.5rem;
}
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
.alias-input textarea:focus, .alias-input input:focus {
    border-bottom-color: var(--accent) !important;
}

/* Character form card */
.form-card {
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 12px;
    padding: 1.75rem;
    position: relative;
}
.form-card label {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.85rem !important;
    color: var(--text-muted) !important;
}
.form-card textarea, .form-card input {
    background: var(--bg) !important;
    border: 1px solid var(--panel-border) !important;
    color: var(--text) !important;
}

/* Avatar popup: a narrower card to read as a distinct interruption */
.popup-card {
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 12px;
    padding: 2rem;
    max-width: 420px;
    margin: 3rem auto;
    text-align: center;
}

/* Buttons: one accent color, used only for the primary action */
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

.field-error {
    color: var(--error) !important;
    font-size: 0.9rem;
}

.avatar-display img {
    border-radius: 12px;
    border: 1px solid var(--panel-border);
}

/* Sidebar */
.sidebar {
    background: var(--panel);
    border-right: 1px solid var(--panel-border);
    padding: 1.5rem 1rem;
}
.sidebar-new-btn, .sidebar-new-btn button {
    width: 100%;
    margin-bottom: 1rem;
}
.sidebar-gallery img {
    border-radius: 8px;
}
"""

with gr.Blocks(title = "Local Character.AI", css = CUSTOM_CSS) as demo:

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

        # Sidebar
        with gr.Column(scale = 1, min_width = 220, elem_classes = "sidebar"):
            new_char_btn = gr.Button(
                "+ New Character", elem_classes="btn-primary sidebar-new-btn"
            )
            sidebar_gallery = gr.Gallery(
                label = "Your Characters",
                columns = 1,
                height = 420,
                show_label = True,
                allow_preview = False,
                elem_classes = "sidebar-gallery",
            )

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
                    with gr.Row():
                        generate_avatar_btn = gr.Button("Generate avatar", elem_classes = "btn-primary")
                        skip_avatar_btn = gr.Button("Skip for now", elem_classes = "btn-secondary")

            # SCREEN 4: chat
            with gr.Column(visible = False, elem_classes = "screen") as screen_chat:
                gr.Markdown("Step 4 of 4", elem_classes = "step-caption")
                greeting = gr.Markdown()
                with gr.Row():
                    with gr.Column(scale = 1):
                        avatar_display = gr.Image(label = "Avatar", elem_classes = "avatar-display")
                    with gr.Column(scale = 2):
                        chatbot_ui = gr.Chatbot(label = "Chat Window")
                        msg_input = gr.Textbox(
                            placeholder = "Say something...", show_label = False
                        )
                        clear_btn = gr.ClearButton([msg_input, chatbot_ui])

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

    def do_generate_avatar(appearance, alias, name, personality, characters):
        """
        do_generate_avatar: Gradio callback for the generate avatar button on the popup screen. Runs Stable Diffusion inference,
        saves the new character (removing the oldest one if max characters), refreshes the sidebar, and then advances to the chat
        screen with an empty hisotyr.

        Input(s):
            appearance: value of char_appearance, used as the image gen prompt
            alias: the user's chosen alias
            name, personality: of the character
            characters: list of existing characters

        Output(s): a tuple of the generated PIL.Image.Image for avatar_display, screen popup's visibility (hidden), screen_chat's visibility (shown), and the greeting Markdown text.
        """

        img = avatar_gen.generate_avatar(appearance)
        avatar_gen.release_gpu_memory()

        # Add the new character, removing the oldest if needed, and populate the sidebar
        characters = list(characters)
        if len(characters) >= MAX_CHARACTERS:
            # Pop oldest to make room
            characters.pop(0)
        characters.append({"name": name, "personality": personality, "appearance": appearance, "avatar": img, "history": []})
        new_index = len(characters) - 1
        gallery_items = [(c["avatar"], c["name"]) for c in characters]
        greeting_text = f"**{alias}**, meet **{name}**. Say hello below!"
        
        return img, gr.update(visible = False), gr.update(visible = True), greeting_text, characters, new_index, gallery_items, []

    generate_avatar_btn.click(
        fn = do_generate_avatar,
        inputs = [char_appearance, user_alias_state, char_name, char_personality, characters_state],
        outputs = [avatar_display, screen_popup, screen_chat, greeting, characters_state, current_index, sidebar_gallery, chatbot_ui],
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

        avatar_gen.release_gpu_memory()
        placeholder = make_placeholder_avatar(name)

        # Again, add the new character and remove oldest if needed. Use a placeholder image for the avatar instead
        characters = list(characters)
        if len(characters) >= MAX_CHARACTERS:
            characters.pop(0)
        characters.append(
            {"name": name, "personality": personality, "appearance": appearance, "avatar": placeholder, "history": []}
        )
        new_index = len(characters) - 1
        gallery_items = [(c["avatar"], c["name"]) for c in characters]
        greeting_text = f"**{alias}**, meet **{name}**. Say hello below!"

        return placeholder, gr.update(visible = False), gr.update(visible = True), greeting_text, characters, new_index, gallery_items, []

    skip_avatar_btn.click(
        fn = skip_avatar,
        inputs = [user_alias_state, char_name, char_personality, char_appearance, characters_state],
        outputs = [avatar_display, screen_popup, screen_chat, greeting, characters_state, current_index, sidebar_gallery, chatbot_ui],
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
        )

    new_char_btn.click (
        fn = start_new_character,
        inputs = [],
        outputs = [char_name, char_personality, char_appearance, current_index,screen_chat, screen_popup, screen_form, form_error],
    )

    def load_character(evt: gr.SelectData, characters):
        """
        load_character: Gradio callback fired when a character is clicked in the sidebar gallery. Loads that character's avatar, 
        saved chat history, name, and personality, then jumps to the chat screen.

        Input(s):
            evt: Gradio's selection event data; evt.index is the clicked character's position in characters.
            characters: the current characters_state list.
        """

        # Use position of the clicked gallery item (character) to look up the full character dict at that pos
        idx = evt.index
        c = characters[idx]

        greeting_text = f"**{c['name']}** is ready to talk. Say hello below!"

        # Return params needed in outputs[]
        return (
            idx,
            c["avatar"], c["history"],
            greeting_text,
            c["name"], c["personality"], c["appearance"],
            gr.update(visible = False), gr.update(visible = False), gr.update(visible = True),
        )

    sidebar_gallery.select(
        fn = load_character,
        inputs = [characters_state],
        outputs = [current_index, avatar_display, chatbot_ui, greeting, char_name, char_personality, char_appearance, screen_form, screen_popup, screen_chat],
    )

    def user_chat(user_msg, alias, characters, idx):
        """
        user_chat: Gradio callback for when the user submits a chat message. It:
        
            1. Sends the message along with running history, current character settings, and the user's alias to the chatbot
            2. Appends the exchange to the visible chat history
            3. Clears the input box
        
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
        # FIRST YIELD: pushes the user's message (and the empty assistant bubble) to the UI
        yield "", display_history, characters

        # bot.respond is the generator from chatbot.py; each yield from it hands back the reply accumulated so far
        for partial_reply in bot.respond(
            user_msg, prior_history, char["name"], char["personality"], user_alias=alias
        ):
            # Overwrite the last item in display_history (the empty or partial assistant bubble) with the latest text
            display_history[-1]["content"] = partial_reply
            # SECOND YIELD: re-render the chat window, acrtually update with new text
            yield "", display_history, characters

        # Ollama message completed, update character's record to preserve history
        updated_char = dict(char)
        updated_char["history"] = display_history
        characters[idx] = updated_char

        # THIRD YIELD: pushes the saved state to the UI one more time, guarsanteeing that the UI reflects the fully saved characters_state, not jsut the local display_history
        yield "", display_history, characters

    msg_input.submit (
        fn = user_chat, 
        inputs = [msg_input, user_alias_state, characters_state, current_index], 
        outputs = [msg_input, chatbot_ui, characters_state]
    )

if __name__ == "__main__":
    demo.launch()