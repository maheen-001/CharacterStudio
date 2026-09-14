# This file was made by Maheen Abbasi on Sep. 11, 2026 as part of a personal project

"""
app.py: A self contained Gradio web app that lets a user define a custom character (name, personality, and visual appearance),
generate an avatar image for that character through StableDiffusion, and then chat with the character via a locally-hosted LLM (Ollama).

Run directly (`python app.py`) to launch the Gradio interface in a browser.
--> Should run locally as long as necessary GPU/CPU resources exist and Ollama is installed with the target model already pulled.
"""

# Imports
import gradio as gr
from image_gen import AvatarGenerator
from chatbot import CharacterChatbot

# Initialize models
avatar_gen = AvatarGenerator()
bot = CharacterChatbot(model_name = "llama3:8b")

"""
STYLING
"""
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500&family=JetBrains+Mono:wght@400;500&display=swap');

.gradio-container {
    --bg: #14161F;
    --panel: #1D2130;
    --panel-border: #2B3044;
    --accent: #E8A33D;
    --accent-soft: rgba(232, 163, 61, 0.15);
    --text: #ECEAE3;
    --text-muted: #9CA3B4;
    --error: #E5786B;

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

/* Small sequence caption ("Step 1 of 4") instead of decorative badges */
.step-caption {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
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
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
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
    color: #14161F !important;
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
"""

with gr.Blocks(title = "Local Character.AI", css = CUSTOM_CSS) as demo:

    # Hold the alias the user picked in screen 1. Threaded through to the chat screen for a personalized greeting and system prompt context.
    user_alias_state = gr.State("")

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

    # SCREEN 2: character creation form
    with gr.Column(visible = False, elem_classes = "screen") as screen_form:
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
                skip_avatar_btn = gr.Button("Skip for now", elem_classes = "btn=secondary")

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

        if not alias or not alias.strip():
            return (
                gr.update(visible = True),
                gr.update(visible = False),
                gr.update(),
                gr.update(value = "Please enter a name to continue.", visible = True),
            )

        return (
            gr.update(visible = False),
            gr.update(visible = True),
            alias.strip(),
            gr.update(visible = False),
        )

    continue_to_form_btn.click(
        fn = submit_alias,
        inputs = [alias_input],
        outputs = [screen_welcome, screen_form, user_alias_state, alias_error],
    )

    # Allow pressif enter in the alias box as well just for ease
    alias_input.submit(
        fn = submit_alias,
        inputs = [alias_input],
        outputs = [screen_welcome, screen_form, user_alias_state, alias_error],
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

    def do_generate_avatar(appearance, alias):
        """
        do_generate_avatar: Gradio callback for the generate avatar button on the popup screen. Runs Stable Diffusion inference,
        then advances to the chat screen with a personalized greeting.

        Input(s):
            appearance: value of char_appearance, used as the image gen prompt
            alias: the user's chosen alias

        Output(s): a tuple of the generated PIL.Image.Image for avatar_display, screen popup's visibility (hidden), screen_chat's visibility (shown), and the greeting Markdown text.
        """

        img = avatar_gen.generate_avatar(appearance)
        avatar_gen.release_gpu_memory()
        greeting_text = f"**{alias}**, your character is ready. Say hello below!"
        return img, gr.update(visible = False), gr.update(visible = True), greeting_text

    generate_avatar_btn.click(
        fn = do_generate_avatar,
        inputs = [char_appearance, user_alias_state],
        outputs = [avatar_display, screen_popup, screen_chat, greeting],
    )

    def skip_avatar(alias):
        """
        skip_avatar: a Gradio callback for the skip for now button on the popup screen (for the avatar gen). Skips straight
        to the chat screen without running image gen, leaving avatar_display empty.

        Input(s):
            alias: the uswr's chosen alias from user_alias_state
        
            Output(s): a tuple of screen_popup's visibility (hidden), screen_chat's visibility(shown), and the greeting Markdown text
        """

        avatar_gen.release_gpu_memory()
        greeting_text = f"**{alias}**, your character is ready. Say hello below!"
        return gr.update(visible = False), gr. update(visible = True), greeting_text

    skip_avatar_btn.click(
        fn = skip_avatar,
        inputs = [user_alias_state],
        outputs = [screen_popup, screen_chat, greeting],
    )

    def user_chat(user_msg, history, name, personality, alias):
        """
        user_chat: Gradio callback for when the user submits a chat message. It:
        
            1. Sends the message along with running history, current character settings, and the user's alias to the chatbot
            2. Appends the exchange to the visible chat history
            3. Clears the input box
        
            Input(s):
                user_msg: The text that the user just typed and submitted
                history: the current Gradio chatbot history
                name: current calue of the char_name textbox
                personality: current calue of the char_personality textbox
                alias: the user's chosen alias from user_alias_state
        """

        bot_reply = bot.respond(user_msg, history, name, personality, user_alias = alias)
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": bot_reply})
        return "", history

    msg_input.submit(
        fn = user_chat, 
        inputs = [msg_input, chatbot_ui, char_name, char_personality, user_alias_state], 
        outputs = [msg_input, chatbot_ui]
    )

if __name__ == "__main__":
    demo.launch()