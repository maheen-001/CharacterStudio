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
bot = CharacterChatbot(model_name = "llama3.2:1b")

with gr.Blocks(title = "Local Character.AI") as demo:
    gr.Markdown("# Local Character Creation & Chat")

    # Filler for character info, will 'force" customization in later update.
    with gr.Row():
        # Left col: character definition (name, personality, and appearance) and the generated avatar preview
        with gr.Column(scale = 1):
            char_name = gr.Textbox(label = "Character Name", value = "Vesper")
            char_personality = gr.Textbox(
                label = "Personality", 
                value = "Cynical cyber-hacker, speaks in brief technical jargon, sarcastic."
            )
            char_appearance = gr.Textbox(
                label = "Appearance Prompt", 
                value = "cyberpunk hacker with glowing neon goggles, dark hoodie"
            )
            create_btn = gr.Button("Create Character")
            avatar_display = gr.Image(label = "Avatar")

        # Right col: the chat interface for talking to the character    
        with gr.Column(scale = 2):
            chatbot_ui = gr.Chatbot(label = "Chat Window")
            msg_input = gr.Textbox(label = "Your Message", placeholder = "Say something...")
            clear_btn = gr.ClearButton([msg_input, chatbot_ui])

    def make_character(appearance):
        """
        make_character: Gradio callbacl for the Create Character button. Basically wires avatar gen.

        Input(s):
            appearance: the text from the char_appearance textbox

        Returns: a PIL.Image.Image of the generated avatar, which is displayed in avatar_display
        """

        img = avatar_gen.generate_avatar(appearance)
        return img

    create_btn.click(fn = make_character, inputs = [char_appearance], outputs = [avatar_display])

    def user_chat(user_msg, history, name, personality):
        """
        user_chat: Gradio callback for when the user submits a chat message. It:
        
            1. Sends the message along with running history and current character settings to the chatbot
            2. Appends the exchange to the visible chat history
            3. Clears the input box
        
            Input(s):
                user_msg: The text that the user just typed and submitted
                history: the current Gradio chatbot history
                name: current calue of the char_name textbox
                personality: current calue of the char_personality textbox
        """

        bot_reply = bot.respond(user_msg, history, name, personality)
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": bot_reply})
        return "", history

    msg_input.submit(
        fn = user_chat, 
        inputs = [msg_input, chatbot_ui, char_name, char_personality], 
        outputs = [msg_input, chatbot_ui]
    )

if __name__ == "__main__":
    demo.launch()