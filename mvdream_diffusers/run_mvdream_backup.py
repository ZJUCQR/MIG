import torch

if torch.cuda.is_available():
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)
    except Exception:
        pass

import kiui
import argparse
from pipeline_mvdream import MVDreamPipeline

pipe = MVDreamPipeline.from_pretrained(
    # "./weights_mvdream", # local weights
    'ashawkey/mvdream-sd2.1-diffusers', # remote weights
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

pipe = pipe.to("cuda")


parser = argparse.ArgumentParser(description="MVDream")
parser.add_argument("prompt", type=str, default="a cute owl 3d model")
args = parser.parse_args()

images = pipe(args.prompt, guidance_scale=5, num_inference_steps=30, elevation=0)

for view_idx, image in enumerate(images):
    kiui.write_image(f"test_mvdream_view_{view_idx}.jpg", image)
