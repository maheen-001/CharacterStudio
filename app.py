# This file was made by Maheen Abbasi on Sep. 11, 2026 as part of a personal project

"""
app.py: A self-contained Gradio web app that walks a user through a 4-step flow:
    1. enter a display profile (alias, plus optional gender/age/context)
    2. build a character via a styled form, including an optional lorebook (worldbuilding notes)
    3. optionally generate an avatar 
    4. chat with the character via a locally-hosted LLM (Ollama), with the avatar generated through Stable Diffusion.

    - A siidebar lists up to MAX_CHARACTERS previously created characters with a button to start building a new one.
    - The bot sends the first message (as an in-character greeting) rather than waiting on the user, and replies stream in by tokens
    with a bouncing dots indicater while Ollama is still warming up and/or generating.
    - Screens are implemented as toggled gr.Column's within a single Blocks layout, so only one is visible at a time, driven by button click callbacks.
    - The sidebar is implemented as MAX_CHARACTERS pre-built rows, toggled visible/hidden as characters are added.
    - Clicking "i" next to the chat avatar opens a READ ONLY character-details popup (personality/appearance/lorebook)
        -> not editable, since editing mid-convo would make the saved history inconsistent with the character's definition.

    Run directly (python app.py on Windows) to launch the Gradio interface in a browser.
        -> Shoudl run locally as long as necessary GPU/CPU resources exist and Ollama is installed with the target model (llama3:8b) already pulled.

"""

# Imports
import os
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
from image_gen import AvatarGenerator
from chatbot import CharacterChatbot, MAX_RECENT_MESSAGES

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

# Default lorebook text for the built-in Baymax example (keeps it consistent with Big Hero 6)
BAYMAX_LOREBOOK = (
    "Setting: San Fransokyo, a fictional fusion of San Francisco and Tokyo. "
    "Baymax was built by Tadashi Hamada as a personal healthcare companion "
    "robot; Tadashi died in a fire at the San Fransokyo Institute of "
    "Technology. Baymax now lives with Tadashi's younger brother, Hiro "
    "Hamada, a robotics prodigy. Baymax is part of a superhero team called "
    "Big Hero 6, alongside Hiro, GoGo Tomago, Wasabi, Honey Lemon, and Fred. "
    "Baymax scans people for injuries and cares for their physical and "
    "emotional wellbeing, and says 'I am satisfied with my care' when a "
    "task is complete."
)

# Cancellation flag for avatar gen / greeting streams. Plain dict on purpose, see module docstring above for why this
# can't just be a gr.State.
_generation_state = {"cancelled": False}

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

def format_character_details(c: dict) -> str:
    """
    format_character_details: builds the read-only Markdown text shown in the character-details popup.

    Input(s):
        c: a character dict (name/personality/appearance/lorebook).

    Output(s): a Markdown-formatted string.
    """
    lorebook = c.get("lorebook", "").strip()
    lore_section = f"\n\n### World notes:\n{lorebook}" if lorebook else ""
    return (
        f"## {c['name']}\n\n"
        f"### Personality:\n{c['personality']}\n\n"
        f"### Appearance:\n{c['appearance']}"
        f"{lore_section}"
    )

def format_user_profile(profile: dict) -> str:
    """
    format_user_profile: builds the read-only Markdown text shown when the user views their own profile
    details (alias/gender/age/context) from the chat screen.

    Input(s):
        profile: the user's profile dict from user_profile_state. Any key besides alias can be empty.

    Output(s): a Markdown-formatted string.
    """
    lines = [f"## {profile.get('alias', 'You')}"]
    if profile.get("gender"):
        lines.append(f"### Gender:\n{profile['gender']}")
    if profile.get("age"):
        lines.append(f"### Age:\n{profile['age']}")
    if profile.get("context"):
        lines.append(f"### Context:\n{profile['context']}")
    if len(lines) == 1:
        # Nothing but the alias was ever filled in
        lines.append("_No other details were added._")
    return "\n\n".join(lines)


def _load_css(filename: str = "style.css") -> str:
    """
    _load_css: reads the external stylesheet from disk next to this file so that
    styling is its own .css file instead of a giant string embedded here.
    """
    css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    with open(css_path, "r", encoding = "utf-8") as f:
        return f.read()

# Set up the css
CUSTOM_CSS = _load_css()


with gr.Blocks(title = "Character Studio") as demo:

    # Full user profile {"alias": str, "gender": str, "age": str, "context": str}. Threaded through to the chat screen for a
    # personalized greeting and system prompt context.
    user_profile_state = gr.State({})

    # List of saved character dicts: {"name", "personality", "appearance", "lorebook", "avatar" (PIL.Image), "history" (list of role/content dicts)}
    characters_state = gr.State([])

    # Index into characters_state for whichever character is currently active in the chat screen (None when building a new one)
    current_index = gr.State(None)

    # True while an avatar (Stable Diffusion) gen or greeting stream is in flight -> gate + new chara and sidebar switches behind a leave confirmation
    sd_generating_state = gr.State(False)
    pending_action_state = gr.State(None)

    # Which character was active before the + New Character was clicked for the back button to work
    previous_index_state = gr.State(None)

    # SCREEN 1: profile entry
    with gr.Column(visible = True, elem_classes = "screen") as screen_welcome:
        gr.Markdown("Step 1 of 4", elem_classes = "step-caption")
        cancel_from_welcome_btn = gr.Button("← Back to chat", elem_classes = "btn-secondary back-btn", visible = False)
        gr.Markdown(
            "## What should we call you here?\nPick any name, it's just for this chat!",
            elem_classes = "welcome-heading",
        )
        alias_input = gr.Textbox(
            placeholder = "Type a name...", show_label = False, elem_classes = "alias-input"
        )
        with gr.Group(elem_classes = "profile-extra"):
            gr.Markdown("Optional: helps the character talk to you naturally.")
            profile_gender = gr.Textbox(label = "Gender (optional)", placeholder = "e.g. she/her, he/him, they/them")
            profile_age = gr.Textbox(label = "Age (optional)", placeholder = "e.g. 24")
            profile_context = gr.Textbox(
                label = "Anything else? (optional)", lines = 2,
                placeholder = "e.g. I'm a college student who loves robotics",
            )
        alias_error = gr.Markdown(visible = False, elem_classes = "field-error")
        continue_to_form_btn = gr.Button("Continue", elem_classes = "btn-primary")

    # MAIN APP: sidebar + the character creation / avatar / chat screens. Hidden until the profile is submitted.
    with gr.Row(visible = False, elem_classes = "app-row") as app_row:

        # Sidebar: + New Character button, then MAX_CHARACTERS pre-built rows (avatar | name), each hidden until a character occupies that slot.
        with gr.Column(scale = 1, min_width = 240, elem_classes = "sidebar") as sidebar_col:
            new_char_btn = gr.Button("+ New Character", elem_classes = "btn-primary sidebar-new-btn")
            gr.Markdown("Chats", elem_classes = "sidebar-header")

            sidebar_rows = []
            for _slot in range(MAX_CHARACTERS):
                with gr.Row(visible = False, elem_classes = "sidebar-row") as _row:
                    with gr.Column(scale = 0, min_width = 40):
                        _avatar = gr.Image(
                            show_label = False, container = False, interactive = False,
                            elem_classes = "sidebar-avatar",
                        )
                    with gr.Column(scale = 1, min_width = 0):
                        _name_btn = gr.Button("", elem_classes = "sidebar-name-btn")
                sidebar_rows.append({"row": _row, "avatar": _avatar, "name_btn": _name_btn})

            # Flattened list of every sidebar component in a fixed order, used whenever a callback needs to refresh the whole sidebar at once.
            sidebar_output_components = []
            for _slot in sidebar_rows:
                sidebar_output_components += [_slot["row"], _slot["avatar"], _slot["name_btn"]]

        # Main content
        with gr.Column(scale = 3, elem_classes = "main-content"):

            # SCREEN 2: character creation form
            with gr.Column(visible = True, elem_classes = "screen") as screen_form:
                gr.Markdown("Step 2 of 4", elem_classes = "step-caption")
                cancel_from_form_btn = gr.Button("← Back to chat", elem_classes = "btn-secondary back-btn", visible = False)
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
                    char_lorebook = gr.Textbox(
                        label = "Lorebook / world notes (optional)",
                        value = BAYMAX_LOREBOOK,
                        lines = 5,
                        placeholder = "Setting, backstory, other characters, rules the character should stay consistent with.",
                    )
                form_error = gr.Markdown(visible = False, elem_classes = "field-error")
                continue_to_avatar_btn = gr.Button("Continue", elem_classes = "btn-primary")

            # SCREEN 3: avatar decision "popup" for generating the image
            with gr.Column(visible = False, elem_classes = "screen") as screen_popup:
                gr.Markdown("Step 3 of 4", elem_classes = "step-caption")
                cancel_from_popup_btn = gr.Button("← Back to chat", elem_classes = "btn-secondary back-btn", visible = False)
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
                    with gr.Column(scale=0, min_width=32):
                        details_btn = gr.Button("i", elem_classes="btn-secondary details-btn")
                    with gr.Column(scale = 0, min_width = 32):
                        my_profile_btn = gr.Button("⚙️", elem_classes="btn-secondary details-btn")

                # sanitize_html = False lets the typing-dots HTML render instead of showing as literal escaped tag text.
                chatbot_ui = gr.Chatbot(show_label = False, elem_classes = "chat-window", sanitize_html = False)
                with gr.Row(elem_classes = "msg-row"):
                    msg_input = gr.Textbox(placeholder = "Message...", show_label = False, elem_classes = "msg_input", scale = 4)
                    send_btn = gr.Button("➤", elem_classes = "send-btn", scale = 0, min_width = 40)
                clear_btn = gr.ClearButton([msg_input, chatbot_ui])

        # Overlay for viewing the chasracter details (read only, no editing)
        with gr.Column(visible = False, elem_classes = "modal-overlay") as screen_char_details:
            with gr.Group(elem_classes = "modal-card"):
                char_details_avatar = gr.Image(
                    show_label = False, container = False, interactive = False,
                    elem_classes = "modal-avatar",
                )
                char_details_text = gr.Markdown(elem_classes = "modal-body", container = False)
                close_details_btn = gr.Button("Close", elem_classes = "btn-secondary modal-close-btn")

        # Overlay for viewing the user's OWN profile details (alias/gender/age/context), accessible from any chat
        with gr.Column(visible = False, elem_classes = "modal-overlay") as screen_my_profile:
            with gr.Group(elem_classes = "modal-card"):
                my_profile_text = gr.Markdown(elem_classes = "modal-body", container = False)
                close_my_profile_btn = gr.Button("Close", elem_classes = "btn-secondary modal-close-btn")

        # OVERLAY: leave-confirmation
        with gr.Column(visible = False, elem_classes = "modal-overlay") as screen_leave_confirm:
            with gr.Group(elem_classes = "modal-card"):
                gr.Markdown("### Still generating")
                gr.Markdown(
                    "This character's avatar or greeting is still being generated. "
                    "Leaving now won't stop the generation in the background, but it "
                    "won't be saved. Leave anyway?"
                )
                with gr.Row():
                    confirm_leave_btn = gr.Button("Leave anyway", elem_classes = "btn-primary")
                    stay_btn = gr.Button("Stay", elem_classes = "btn-secondary")

    """
    Helpers
    """

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
        char_appearance, char_lorebook, screen_form visibility, screen_popup visibility, screen_chat visibility).
        """
        c = characters[idx]

        return (
            idx,
            c["avatar"], c["history"], f"**{c['name']}**",
            c["name"], c["personality"], c["appearance"], c.get("lorebook", ""),
            gr.update(visible = False), gr.update(visible = False), gr.update(visible = True),
        )

    def start_new_character_updates():
        """
        start_new_character_updates: the field/screen updates for jumping
        back to the character-creation screen with blank fields.
        """
        return (
            "", "", "", "",  # name, personality, appearance, lorebook
            None,  # current_index
            gr.update(visible = False), gr.update(visible = False),
            gr.update(visible = True), gr.update(visible = False),
            gr.update(interactive = True), gr.update(interactive = True), gr.update(visible = False),
        )

    def start_new_character_profile_updates():
        """
        start_new_character_profile_updates: clears the temporary profile fields
        and returns the screen updates for starting a new character from the profile step.
        """
        return (
            # screen_welcome, app_row
            gr.update(visible = True), gr.update(visible = False),
            # user_profile_state and alias_error
            {}, gr.update(visible = False),
        )
    
    # ----------------------------------------------------------------
    # Callbacks: profile entry
    # ----------------------------------------------------------------

    def submit_profile(alias, gender, age, context):
        """
        submit_profile: Gradio callback for continuing past the profile screen.
        Only alias is required; gender/age/context are optional and stored
        as-is (empty string if left blank).
        """
        if not alias or not alias.strip():
            return (
                gr.update(visible = True), gr.update(visible = False),
                gr.update(),
                gr.update(value = "Please enter a name to continue.", visible = True),
                gr.update(),
            )
        profile = {
            "alias": alias.strip(),
            "gender": (gender or "").strip(),
            "age": (age or "").strip(),
            "context": (context or "").strip(),
        }
        return (
            gr.update(visible = False), gr.update(visible = True),
            profile,
            gr.update(visible = False), gr.update(visible = False),
        )

    # Wire button
    continue_to_form_btn.click(
        fn = submit_profile,
        inputs = [alias_input, profile_gender, profile_age, profile_context],
        outputs = [screen_welcome, app_row, user_profile_state, alias_error, sidebar_col],
    )
    alias_input.submit(
        fn = submit_profile,
        inputs = [alias_input, profile_gender, profile_age, profile_context],
        outputs = [screen_welcome, app_row, user_profile_state, alias_error, sidebar_col],
    )

    # ----------------------------------------------------------------
    # Callbacks: character form
    # ----------------------------------------------------------------

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

    # Wire button
    continue_to_avatar_btn.click(
        fn = submit_character,
        inputs = [char_name, char_personality, char_appearance],
        outputs = [screen_form, screen_popup, form_error],
    )

    # ----------------------------------------------------------------
    # Callbacks: avatar generation + greeting (cancellable)
    # ----------------------------------------------------------------

    def _finish_character_creation(new_char, profile, characters):
        """
        _finish_character_creation: shared logic for both do_generate_avatar and skip_avatar. 
        Saves the new character (popping the oldest if at MAX_CHARACTERS), reveals the chat screen, and streams 
        an in-character greeting as the first chat message. Checks the _generation_state["cancelled"] flag at each
        loop iteration, and stops without saving the character if it's true.

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
        for partial_greeting in bot.greet(
            new_char["name"], new_char["personality"],
            char_lorebook = new_char.get("lorebook"), user_profile = new_char.get("profile", {}),
            memory_summary = new_char.get("memory_summary", ""),
        ):
            display_history[-1]["content"] = partial_greeting
            yield (
                gr.update(), gr.update(),
                gr.update(), gr.update(),
                characters, new_index, display_history,
                *no_op_sidebar_updates(), gr.update(),
            )

        # Check for cancellation
        if _generation_state["cancelled"]: return

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

    def do_generate_avatar(appearance, profile, name, personality, lorebook, characters):
        """
        do_generate_avatar: Gradio callback for the generate avatar button on the popup screen. Disables bth popup buttons,
        shows a loading message, then runs Stable Diffusion inference before handing off to _finish_character_creation for saving + the streamed greeting.
        Also checks if user left mid-generation, and discards the image when it is made rather than saving it.

        Input(s):
            appearance: value of char_appearance, used as the image gen prompt
            profile: the user's profile dict (alias/gender/age/context)
            name, personality, lorebook: of the character
            characters: list of existing characters

        Output(s): a tuple of the generated PIL.Image.Image for avatar_display, screen popup's visibility (hidden), screen_chat's visibility (shown), and the greeting Markdown text.
        """
        _generation_state["cancelled"] = False

        # FIRST YIELD: locks the popup before starting the SD call (everyting unrelated to the lock is just gr.update() since ntohing else should change yet)
        yield (
            gr.update(), gr.update(), gr.update(), gr.update(),
            characters, gr.update(), gr.update(),
            *no_op_sidebar_updates(), gr.update(),

            # Lock generate and skip avatar buttons
            gr.update(interactive = False), gr.update(interactive = False),

            # Show loading popup
            gr.update(visible = True),

            # sd_generating_state -> True, a generation is now in flight
            True,
            gr.update(),
        )

        img = avatar_gen.generate_avatar(appearance)
        avatar_gen.release_gpu_memory()

        # Check if the user left during generation
        if _generation_state["cancelled"]:
            _generation_state["cancelled"] = False
            return

        new_char = {"name": name, "personality": personality, "appearance": appearance, "lorebook": lorebook, "avatar": img, "history": [], "profile": profile, "memory_summary": "", "summarized_through": 0}

        for step in _finish_character_creation(new_char, profile, characters):
            yield (*step, gr.update(interactive = True), gr.update(interactive = True), gr.update(visible = False), False, gr.update(visible = True))

    # Wire button
    generate_avatar_btn.click(
        fn = do_generate_avatar,
        inputs = [char_appearance, user_profile_state, char_name, char_personality, char_lorebook, characters_state],
        outputs = [avatar_display, chat_header_name, screen_popup, screen_chat, characters_state, current_index, chatbot_ui, *sidebar_output_components, msg_input, generate_avatar_btn, skip_avatar_btn, popup_loading, sd_generating_state, sidebar_col],
    )

    def skip_avatar(profile, name, personality, appearance, lorebook, characters):
        """
        skip_avatar: a Gradio callback for the skip for now button on the popup screen (for the avatar gen). Same as do+generate_avatar,
        but uses a generated placeholder image instead of running Stable Diffusion.

        Input(s):
            profile: the user's profile dict (alias/gender/age/context)
            name, personality, appearance, lorebook: of the created character
            characters: list of existing characters
        
            Output(s): a tuple of screen_popup's visibility (hidden), screen_chat's visibility(shown), and the greeting Markdown text
        """

        _generation_state["cancelled"] = False

        # FIRST YIELD
        yield (
            gr.update(), gr.update(), gr.update(), gr.update(),
            characters, gr.update(), gr.update(),
            *no_op_sidebar_updates(), gr.update(),
        
            # Lock generate and skip avatar buttons
            gr.update(interactive = False), gr.update(interactive = False),
        
            # Show loading popup
            gr.update(visible = True),

            # sd_generating_state -> True
            True,
            gr.update(),
        )
        
        placeholder = make_placeholder_avatar(name)
        avatar_gen.release_gpu_memory()

        # Check if user left during generation
        if _generation_state["cancelled"]:
            _generation_state["cancelled"] = False
            return
        
        new_char = {"name": name, "personality": personality, "appearance": appearance, "lorebook": lorebook, "avatar": placeholder, "history": [], "profile": profile, "memory_summary": "", "summarized_through": 0}

        for step in _finish_character_creation(new_char, profile, characters):
            yield (*step, gr.update(interactive = True), gr.update(interactive = True), gr.update(visible = False), False, gr.update(visible = True))


    # Wire button
    skip_avatar_btn.click(
        fn = skip_avatar,
        inputs = [user_profile_state, char_name, char_personality, char_appearance, char_lorebook, characters_state],
        outputs = [avatar_display, chat_header_name, screen_popup, screen_chat, characters_state, current_index, chatbot_ui, *sidebar_output_components, msg_input, generate_avatar_btn, skip_avatar_btn, popup_loading, sd_generating_state, sidebar_col],
    )

    # ----------------------------------------------------------------
    # Callbacks: navigation gated behind still generating check
    # ----------------------------------------------------------------

    def handle_new_character_click(generating, current_idx):
        """
        handle_new_character_click: if a generation is in flight, shows the
        leave-confirmation overlay instead of navigating immediately;
        otherwise proceeds straight to a blank character form.
        """
        if generating:
            return (
                "new_character",                      # pending_action_state
                gr.update(visible=True),              # screen_leave_confirm

                gr.update(),                          # char_name
                gr.update(),                          # char_personality
                gr.update(),                          # char_appearance
                gr.update(),                          # char_lorebook

                gr.update(),                          # current_index
                gr.update(),                          # screen_chat
                gr.update(),                          # screen_popup
                gr.update(),                          # screen_form
                gr.update(),                          # form_error
                gr.update(),                          # generate_avatar_btn
                gr.update(),                          # skip_avatar_btn
                gr.update(),                          # popup_loading

                gr.update(),                          # screen_welcome
                gr.update(),                          # app_row
                gr.update(),                          # user_profile_state
                gr.update(),                          # alias_error
                gr.update(),                          # sidebar_col

                gr.update(),                          # previous_index_state
                gr.update(),                          # cancel_from_welcome_btn
                gr.update(),                          # cancel_from_form_btn
                gr.update(),                          # cancel_from_popup_btn
            )

        char_blanks = start_new_character_updates()
        profile_blanks = start_new_character_profile_updates()

        has_previous = current_idx is not None
        return (None, gr.update(visible = False), *char_blanks, *profile_blanks, gr.update(visible = False), current_idx, gr.update(visible = has_previous), gr.update(visible = has_previous), gr.update(visible = has_previous))    

    # Wire button
    new_char_btn.click(
        fn = handle_new_character_click,
        inputs = [sd_generating_state, current_index],
        outputs = [pending_action_state, screen_leave_confirm, char_name, char_personality, char_appearance, char_lorebook, current_index, screen_chat, screen_popup, screen_form, form_error, generate_avatar_btn, skip_avatar_btn, popup_loading, screen_welcome, app_row, user_profile_state, alias_error, sidebar_col, previous_index_state, cancel_from_welcome_btn, cancel_from_form_btn, cancel_from_popup_btn],
    )

    # Loop through every character slot in the sidebar and set up a click handler for each one
    for i, slot in enumerate(sidebar_rows):
        def handle_sidebar_click(characters, generating, slot_index = i):
            """
            handle_sidebar_click: same generation gate pattern as handle_new_character_click, but for switching to an existing
            saved character instead of starting a blank one.
            """
            if generating:
                return (
                    slot_index, gr.update(visible = True),
                    gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(),
                )
            loaded = load_character_by_index(slot_index, characters)
            # loaded = (idx, avatar, history, header_name, name, personality, appearance, lorebook, form_vis, popup_vis, chat_vis)
            return (None, gr.update(visible = False), *loaded)

        slot["name_btn"].click(
            fn = handle_sidebar_click,
            inputs = [characters_state, sd_generating_state],
            outputs = [pending_action_state, screen_leave_confirm, current_index, avatar_display, chatbot_ui, chat_header_name, char_name, char_personality, char_appearance, char_lorebook, screen_form, screen_popup, screen_chat],
        )

    def confirm_leave(pending_action, characters):
        """
        confirm_leave: "Leave anyway" on the confirmation overlay. Sets the cancellation flag so the 
        generator discards its result, then performs whichever navigation was pending.
        """
        _generation_state["cancelled"] = True

        if pending_action == "new_character":
            blanks = start_new_character_updates()
            # blanks = (name, personality, appearance, lorebook, idx, chat_vis, popup_vis, form_vis, form_error, gen_btn, skip_btn, loading)
            return (
                gr.update(visible = False), False,
                blanks[4],  # current_index
                blanks[5], blanks[6], blanks[7], # screen_chat, screen_popup, screen_form
                gr.update(), gr.update(), gr.update(), # avatar/chatbot/header unchanged
                blanks[0], blanks[1], blanks[2], blanks[3], # name/personality/appearance/lorebook
                blanks[9], blanks[10], blanks[11], # gen_btn, skip_btn, loading
            )
        else:
            idx = pending_action
            loaded = load_character_by_index(idx, characters)
            # loaded = (idx, avatar, history, header_name, name, personality, appearance, lorebook, form_vis, popup_vis, chat_vis)
            return (
                gr.update(visible = False), False,
                loaded[0],
                loaded[10], loaded[9], loaded[8], # screen_chat, screen_popup, screen_form
                loaded[1], loaded[2], loaded[3], # avatar, chatbot, header_name
                loaded[4], loaded[5], loaded[6], loaded[7], # name/personality/appearance/lorebook
                gr.update(interactive = True), gr.update(interactive = True), gr.update(visible = False),
            )

    # Wire button
    confirm_leave_btn.click(
        fn = confirm_leave,
        inputs = [pending_action_state, characters_state],
        outputs = [screen_leave_confirm, sd_generating_state, current_index, screen_chat, screen_popup, screen_form, avatar_display, chatbot_ui, chat_header_name, char_name, char_personality, char_appearance, char_lorebook, generate_avatar_btn, skip_avatar_btn, popup_loading],
    )

    def stay():
        """
        stay: "Stay" on the confirmation overlay (just hide it, nothing else changes).
        """
        return gr.update(visible = False)

    def cancel_character_creation(previous_index, characters):
        """
        cancel_character_creation: Gradio callback for the "← Back to chat" button, shown on the profile,
        character form, and avatar popup screens once there's already at least one character. Lets the user
        bail out of creating a new character and jump straight back to whichever chat they came from.

        Input(s):
            previous_index: index into characters, saved by handle_new_character_click right before it
                cleared current_index to build a new one.
            characters: the current characters_state list.
        """
        if previous_index is None or previous_index >= len(characters):
            # Shouldn't normally happen (the button is only shown when there IS a previous character)
            return (
                gr.update(visible = False), gr.update(visible = True), gr.update(visible = False),
                gr.update(visible = True), gr.update(visible = False), gr.update(visible = False),
                gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(),
            )

        loaded = load_character_by_index(previous_index, characters)
        # loaded = (idx, avatar, history, header_name, name, personality, appearance, lorebook, form_vis, popup_vis, chat_vis)
        return (
            gr.update(visible = False),  # screen_welcome: hide, we're heading back to the chat
            gr.update(visible = True),   # app_row: reveal (sidebar + main content)
            gr.update(visible = True),   # sidebar_col: reveal, we're back on the chat screen
            loaded[8], loaded[9], loaded[10],  # screen_form, screen_popup, screen_chat
            loaded[0], loaded[1], loaded[2], loaded[3],  # current_index, avatar_display, chatbot_ui, chat_header_name
            loaded[4], loaded[5], loaded[6], loaded[7],  # char_name, char_personality, char_appearance, char_lorebook
        )

    # Wire all three Back buttons to the same handler
    for cancel_btn in [cancel_from_welcome_btn, cancel_from_form_btn, cancel_from_popup_btn]:
        cancel_btn.click(
            fn = cancel_character_creation,
            inputs = [previous_index_state, characters_state],
            outputs = [
                screen_welcome, app_row, sidebar_col, screen_form, screen_popup, screen_chat,
                current_index, avatar_display, chatbot_ui, chat_header_name,
                char_name, char_personality, char_appearance, char_lorebook,
            ],
        )

    # Wire button
    stay_btn.click(fn = stay, inputs = [], outputs = [screen_leave_confirm])

    # ----------------------------------------------------------------
    # Callbacks: character details (read-only)
    # ----------------------------------------------------------------

    def show_details(idx, characters):
        """
        show_details: Gradio callback for the "i" button in the chat header.
        """
        if idx is None or idx >= len(characters):
            return gr.update(), gr.update(), gr.update(visible = False)
        c = characters[idx]
        return c["avatar"], format_character_details(c), gr.update(visible = True)

    # Wire button
    details_btn.click(
        fn = show_details, inputs = [current_index, characters_state],
        outputs = [char_details_avatar, char_details_text, screen_char_details],
    )

    def close_details():
        """
        close_details: hides the character-details overlay.
        """
        return gr.update(visible = False)

    # Wire button
    close_details_btn.click(fn = close_details, inputs = [], outputs = [screen_char_details])

    def show_my_profile(profile):
        """
        show_my_profile: Gradio callback for the "⚙️" button in the chat header, lets the user check what
        profile details they entered for themselves without leaving the current chat.
        """
        return format_user_profile(profile), gr.update(visible = True)

    # Wire button
    my_profile_btn.click(
        fn = show_my_profile, inputs = [user_profile_state],
        outputs = [my_profile_text, screen_my_profile],
    )

    def close_my_profile():
        """
        close_my_profile: hides the my-profile overlay.
        """
        return gr.update(visible = False)

    # Wire button
    close_my_profile_btn.click(fn = close_my_profile, inputs = [], outputs = [screen_my_profile])

    # ----------------------------------------------------------------
    # Callbacks: chat
    # ----------------------------------------------------------------

    def user_chat(user_msg, profile, characters, idx):
        """
        user_chat: Gradio callback for when the user submits a chat message. It:
        
            1. Sends the message along with running history, current character settings, and the user's profile to the chatbot
            2. Appends the exchange to the visible chat history
            3. Clears the input box
        msg_input (ability to send messages) is blocked suring generation to prevent sedning a seconf message mid-reply.
        
            Input(s):
                user_msg: The text that the user just typed and submitted
                profile: the user's profile dict from user_profile_state
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
            {"role": "assistant", "content": TYPING_INDICATOR_HTML},
        ]

        # FIRST YIELD: pushes the user's message (and the empty assistant bubble) to the UI; disables input
        yield gr.update(value = "", interactive = False), display_history, characters

        # bot.respond is the generator from chatbot.py; each yield from it hands back the reply accumulated so far
        for partial_reply in bot.respond(
            user_msg, prior_history, char["name"], char["personality"], char_lorebook = char.get("lorebook"), user_profile = char.get("profile", {}),
            memory_summary = char.get("memory_summary", ""),
        ):
            # Overwrite the last item in display_history (the empty or partial assistant bubble) with the latest text
            display_history[-1]["content"] = partial_reply
            # SECOND YIELD: re-render the chat window, acrtually update with new text
            yield gr.update(), display_history, characters

        # Ollama message completed, update character's record to preserve history
        updated_char = dict(char)
        updated_char["history"] = display_history

        # Implement simple memory: once convo grows past the varbatim window, fold whatever just aged out of it into the running memory_summary so tha future
        # turns stay aware of it without needing to resend the entire history each time
        
        # Find lengths
        already_summarized = updated_char.get("summarized_through", 0)
        keep_verbatim = len(display_history) - MAX_RECENT_MESSAGES

        # New messages outside of the verbatim window -> summarize them
        if keep_verbatim > already_summarized:
            turns_to_fold = display_history[already_summarized:keep_verbatim]
            if turns_to_fold:
                updated_char["memory_summary"] = bot.summarize(
                    turns_to_fold, char["name"], existing_summary = updated_char.get("memory_summary", "")
            )
                updated_char["summarized_through"] = keep_verbatim

        # Save the updated character
        characters[idx] = updated_char

        # THIRD YIELD: pushes the saved state to the UI one more time, guarsanteeing that the UI reflects the fully saved characters_state, not jsut the local display_history
        #    -> Also re-enable input
        yield gr.update(interactive = True), display_history, characters

    msg_input.submit (
        fn = user_chat, 
        inputs = [msg_input, user_profile_state, characters_state, current_index], 
        outputs = [msg_input, chatbot_ui, characters_state]
    )

    # Same callback wired to the send utton so it does the same thing as pressinf enter
    send_btn.click (
        fn = user_chat, 
        inputs = [msg_input, user_profile_state, characters_state, current_index], 
        outputs = [msg_input, chatbot_ui, characters_state]
    )

if __name__ == "__main__":
    demo.launch(css = CUSTOM_CSS)