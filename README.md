# Character Studio

A Python/Gradio web application for creating and chatting with custom AI roleplay characters using a locally hosted language model and optional locally generated character avatars.

The project combines a multi-step graphical interface with local LLM inference, text-to-image generation, conversation memory, SQLite persistence, and customizable character profiles.

> **Project status:** Completed educational / portfolio project

## Overview

Character Studio allows users to create custom roleplay characters and have ongoing conversations with them through a locally hosted AI model.

The application provides a workflow for:

1. Creating a user profile with an alias and optional personal context
2. Building a custom character with a name, personality, appearance, and optional worldbuilding notes
3. Generating a random character using the Surprise Me feature
4. Generating a character avatar using Stable Diffusion
5. Using a placeholder avatar when image generation is skipped
6. Automatically generating an in-character opening message
7. Carrying out streaming conversations with the character
8. Maintaining recent conversation history and longer-term memory summaries
9. Saving characters and conversations locally using SQLite
10. Switching between multiple saved characters through a sidebar
11. Viewing character details and the current user profile during a conversation
12. Deleting saved characters when they are no longer needed

The application is designed as a local educational and portfolio project exploring the integration of generative AI models with a Python-based user interface.

## Technologies

* **Python**
* **Gradio**
* **Ollama**
* **Llama 3 8B**
* **Stable Diffusion v1.5**
* **PyTorch**
* **Hugging Face Diffusers**
* **Transformers**
* **Accelerate**
* **Pillow**
* **SQLite**
* **HTML/CSS**
* **Git / GitHub**

## AI Models

### Language Model

Character Studio uses **Llama 3 8B** through Ollama for character generation and conversation.

The model is used for:

* Generating random characters
* Creating in-character opening messages
* Generating responses during conversations
* Summarizing older conversation history into persistent memory

The application communicates with Ollama locally rather than relying on a remote language-model API.

### Image Generation

Character avatars are generated using **Stable Diffusion v1.5** through the Hugging Face Diffusers library.

The application automatically selects the available compute device:

* NVIDIA CUDA GPU with FP16 when available
* CPU with FP32 as a fallback

Generated avatars are displayed in the chat header, character sidebar, and character-details popup.

If the user chooses not to generate an avatar, Character Studio creates a simple placeholder avatar using the character's initial instead.

## How It Works

### 1. User Profile

The application begins by asking the user to choose a display alias.

The user can optionally provide:

* Gender
* Age
* Additional context

These details are stored as part of the user's profile and are provided to the character as context during conversations.

The profile is optional beyond the alias and can be viewed from the chat interface.

### 2. Character Creation

The character creation screen allows the user to define:

* Character name
* Personality
* Appearance
* Lorebook / world notes
* Optional custom opening message

The lorebook can contain information such as:

* Setting
* Backstory
* Other characters
* Worldbuilding
* Rules the character should remain consistent with

The character's personality, appearance, and lorebook are incorporated into the language model's system prompt.

### 3. Random Character Generation

The **Surprise Me** button generates a character automatically using the local Llama model.

A random archetype and gender are selected first. Example archetypes include:

* Best friend
* Anti-hero
* Trickster
* Mentor
* Flirtatious character
* Awkward nerd
* Fantasy character
* Sci-fi character
* Villain
* Shy character

The selected hints are provided to the language model, which returns a character name, personality, appearance, and lorebook information.

The generated information is then inserted directly into the character creation form for the user to review or modify.

### 4. Avatar Generation

After creating a character, the user can choose whether to generate an avatar.

If avatar generation is selected, the character's appearance description is passed to Stable Diffusion using a predefined portrait prompt.

The generated image is then used as the character's avatar throughout the application.

Alternatively, the user can skip avatar generation and use an automatically generated placeholder containing the character's initial.

### 5. Character Greeting

Once character creation is complete, the character sends the first message automatically rather than waiting for the user.

The greeting can either be:

* A custom opening message written by the user
* A newly generated message from the local language model

The generated greeting is streamed into the interface so that the user can see the response being produced.

### 6. Conversation

Users can then chat with the character through the main chat interface.

The character's responses are generated using:

* Character name
* Character personality
* Character lorebook
* User profile
* Recent conversation history
* Long-term conversation memory
* An example conversation used to guide response formatting

Responses are streamed from Ollama rather than waiting for the complete response before displaying anything.

A typing indicator is shown while the response is being generated.

### 7. Conversation Memory

Character Studio uses two levels of conversation context.

#### Recent History

The most recent messages are passed directly to Ollama for each response.

The current implementation sends up to:

```text
8 recent messages
```

#### Memory Summary

Older conversation turns are periodically condensed into a short summary.

This allows longer conversations to retain important information without continuously sending the entire conversation history to the language model.

The summary prioritizes details such as:

* Names
* Decisions
* Promises
* Important events
* Relationship changes
* Other concrete information established during the conversation

## Character Persistence

Characters are stored locally using SQLite.

Each saved character can contain:

* Name
* Personality
* Appearance
* Lorebook
* Avatar
* Conversation history
* User profile
* Memory summary
* Conversation-summary position
* Last active timestamp

This allows characters and their conversations to persist across browser refreshes and application restarts.

The application can store multiple characters simultaneously and provides a sidebar for switching between them.

The sidebar currently supports up to:

```text
20 characters
```

When the maximum is reached, the oldest saved character is removed to make room for the new one.

## User Interface

The interface is divided into several stages.

### Character Creation Flow

```text
User Profile
     │
     ▼
Character Creation
     │
     ├──► Manual Character
     │
     └──► Surprise Me
              │
              ▼
       Random Character
     │
     ▼
Avatar Decision
     │
     ├──► Generate with Stable Diffusion
     │
     └──► Skip / Placeholder Avatar
     │
     ▼
Character Greeting
     │
     ▼
Chat
```

### Chat Interface

The chat screen contains:

* Character avatar
* Character name
* Character details button
* User profile button
* Conversation history
* Typing indicator
* Message input
* Send button
* Saved-character sidebar

Character details are displayed in a read-only popup so that the character definition remains consistent with the conversation history.

## Project Structure

```text
character-studio/
│
├── app.py
├── chatbot.py
├── db.py
├── image_gen.py
├── style.css
├── requirements.txt
├── README.md
└── ...
```

### Main Files

**`app.py`**

Contains the main Gradio application and coordinates the character creation workflow, UI state, callbacks, chat interface, avatar generation, and character management.

**`chatbot.py`**

Handles communication with the locally hosted Ollama model, including character responses, greetings, random character generation, and conversation-memory summaries.

**`image_gen.py`**

Contains the `AvatarGenerator` class, which loads Stable Diffusion and generates character avatar images.

**`db.py`**

Handles SQLite persistence for saved characters, including inserting, updating, loading, and deleting characters.

**`style.css`**

Contains the custom CSS used to style the Gradio interface, including the sidebar, forms, chat bubbles, modals, buttons, avatars, animations, and overall colour scheme.

**`requirements.txt`**

Lists the Python packages required to run the application.

## Example Workflow

```text
User Profile
      │
      ▼
Choose Alias & Optional Profile Information
      │
      ▼
Build Character
      │
      ├──► Enter Character Manually
      │
      └──► Surprise Me
              │
              ▼
        Generate Character
        with Local LLM
      │
      ▼
Character Information
(Name / Personality / Appearance / Lorebook)
      │
      ▼
Choose Avatar
      │
      ├──► Stable Diffusion
      │
      └──► Placeholder Avatar
      │
      ▼
Generate Opening Message
      │
      ▼
Chat with Character
      │
      ├──► Recent Conversation History
      │
      ├──► Memory Summary
      │
      └──► User Profile
      │
      ▼
Save Character + Conversation
to SQLite
```

## Local Setup

The application is designed to run locally and requires the relevant Python dependencies and AI models to be available on the user's machine.

Install the Python dependencies listed in `requirements.txt`.

The application also requires:

* Ollama
* The `llama3:8b` model
* Sufficient CPU/GPU resources for the selected models
* Stable Diffusion model files/download access for avatar generation

The application can then be launched with:

```bash
python app.py
```

The Gradio interface will open in a web browser.

## Current Limitations

This project has several limitations related primarily to local AI model performance and hardware requirements:

* Character responses depend on the capabilities and behaviour of the locally hosted Llama 3 8B model.
* Response generation can take time depending on available hardware.
* Stable Diffusion avatar generation can be computationally expensive, particularly when running on CPU.
* Generated character information and dialogue are not guaranteed to be factually or narratively consistent.
* The conversation memory system uses summaries rather than retaining the complete conversation in every language-model request.
* The application currently uses a fixed recent-message window when constructing prompts.
* Avatar generation uses a predefined prompting style rather than allowing complete control over image-generation parameters.
* Character editing is intentionally not supported after a conversation has been established, since changing the character definition mid-conversation could make the saved history inconsistent.
* The application is designed primarily for local educational and experimentation purposes rather than production deployment.

## What I Learned

This project has given me experience with:

* Building a multi-step Python/Gradio application
* Designing interactive interfaces with Gradio components and callbacks
* Managing application state across multiple screens
* Integrating a locally hosted LLM through Ollama
* Working with streaming model responses
* Designing prompts for consistent roleplay behaviour
* Generating structured JSON responses from an LLM
* Integrating Stable Diffusion through Hugging Face Diffusers
* Working with PyTorch and GPU/CPU device selection
* Managing conversation history and summarization
* Designing a lightweight long-term memory system
* Persisting application data using SQLite
* Serializing Python objects with `pickle`
* Creating custom UI styling with CSS
* Building responsive sidebar and modal interfaces
* Managing generation workflows through Gradio generators
* Handling loading states and UI controls during model generation
* Organizing a larger application across multiple Python modules
* Using Git and GitHub for version control

## Disclaimer

This is an educational and portfolio project demonstrating the integration of local generative AI models with a Python-based interface.

The generated characters, dialogue, and images are produced by AI models and may contain inaccuracies, unexpected content, or inconsistencies. The application should be treated as an experimental/educational project rather than a production-grade conversational AI system.
