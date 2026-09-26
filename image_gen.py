# This file was made by Maheen Abbasi on Sep. 11, 2026 as part of a personal project

# Imports
import torch 
from diffusers import StableDiffusionPipeline
from PIL import Image

class AvatarGenerator:
    """
    AvatarGenerator: This class wraps a Stable Diffusion v1.5 text to image pipeline for generating
    character avatar portraits from a short visual desc provided by the user.
    --> Loads the model once at construction time (on GPU if available) so that repeated calls to generate_avatar
    don't pay the model loading cost each time (it significantly slows things down).
    """

    def __init__(self):
        """
        Set up the compute device and load the Stable Diffusion pipeline.

        1. CUDA (fp16) IF an NVIDIA GPU is available
        2. Fallback: CPU (fp32)
        """

        # CUDA for Nvidia GPU, otherwise default to cpu
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        # Download and load Stable Diffusion v1.5 weights, then move the pipeline to the selected device
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype = dtype
        ).to(self.device)

    def generate_avatar(self, visual_desc: str) -> Image.Image:
        """
        generate_avatar: This function will generate a single avatar portrait image from a text description.

        Input(s):
            visual_desc: A string with the user's description of the character's appearance; wrapped in a fixed
            prompt template to biad the output towards a more detailed portrait style avatar (highly detailed chara avatar 8k)
        """

        # Just gonna disable the safety check cuz it's a little trigger-happy and it's not like I'm generating any of that...
        self.pipe.safety_checker = None
        self.pipe.requires_safety_checker = False

        prompt = f"drawn portrait of {visual_desc}, attractive adult character, striking features, expressive eyes, stylish appearance, dark romance aesthetic, moody cinematic lighting, elegant fashion, mysterious expression, Pinterest-inspired digital illustration, semi-realistic anime style, clean lineart, detailed shading"

        negative_prompt = """
        photorealistic, photograph, live action, 3d render, hyperrealistic,
        blurry, low quality, poorly drawn face, deformed face, asymmetrical eyes,
        bad anatomy, extra fingers, malformed hands, distorted features,
        oversaturated, childish style
        """

        # CAN CHANGE THESE PARAMS: Using 25 steps for now (testing + efficiency); return the first image in the pipeline's output batch
        image = self.pipe(prompt, negative_prompt = negative_prompt, num_inference_steps = 25).images[0]
        return image

    def release_gpu_memory(self):
        """
        release_gpu_memory: moves the pipeline off the GPU and clears the CUDA cache once avatar gen is no longer needed for this session (chatting started)
        """

        if self.device == "cuda":
            self.pipe.to("cpu")
            torch.cuda.empty_cache()