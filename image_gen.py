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

        prompt = f"portrait photo of {visual_desc}, highly detailed, character avatar, 8k"

        # CAN CHANGE THESE PARAMS: Using 25 steps for now (testing + efficiency); return the first image in the pipeline's output batch
        image = self.pipe(prompt, num_inference_steps = 25).images[0]
        return image