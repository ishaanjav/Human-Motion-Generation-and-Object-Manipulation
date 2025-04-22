import sys
sys.path.append(sys.path[0] + r"/../")
import torch
import lightning as L
from datetime import datetime
import os.path
from os.path import join as pjoin
from models import *
from models.mlp_head import MLPHead
from configs import get_config
from collections import OrderedDict

class LitMLPInferModel(L.LightningModule):
    def __init__(self, intergen_model, mlp_head, cfg):
        super().__init__()
        self.cfg = cfg
        self.intergen_model = intergen_model
        self.mlp_head = mlp_head
        
        # Freeze both models
        self.intergen_model.eval()
        self.mlp_head.eval()
        
        self.normalizer = MotionNormalizer()

    def generate_one_sample(self, prompt, name):
        batch = OrderedDict({})
        batch["motion_lens"] = torch.zeros(1,1).long().cuda()
        batch["text"] = [prompt]
        
        # Get InterGen output
        with torch.no_grad():
            intergen_output = self.intergen_model.forward_test(batch)
            motion_output = intergen_output["output"]  # Shape: B, T, nfeats*2
            
            # Split InterGen output into first and last 256 dimensions
            first_half = motion_output[..., :256]  # First 256 dimensions
            second_half = motion_output[..., 256:]  # Last 256 dimensions
            
            # Process first half through MLP
            mlp_output = self.mlp_head(first_half)
            
            # Concatenate MLP output with untouched second half
            final_output = torch.cat([mlp_output, second_half], dim=-1)
            
        # Process and save results
        result_path = os.path.join("results", "mlp_outputs", f"{name}.npy")
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        torch.save(final_output.cpu(), result_path)
        
        return final_output

def build_models(cfg):
    # Load InterGen model
    intergen_model = InterGen(cfg)
    if cfg.CHECKPOINT:
        ckpt = torch.load(cfg.CHECKPOINT, map_location="cpu")
        intergen_model.load_state_dict(ckpt["state_dict"], strict=False)
        print("InterGen checkpoint loaded!")
    
    # Create MLP head that processes first 256 dimensions
    mlp_head = MLPHead(
        input_dim=256,  # Process first 256 dimensions
        hidden_dims=[1024, 512],
        output_dim=256  # Output 256 dimensions
    )
    
    return intergen_model, mlp_head

if __name__ == '__main__':
    model_cfg = get_config("configs/model.yaml")
    infer_cfg = get_config("configs/infer.yaml")

    # Build models
    intergen_model, mlp_head = build_models(model_cfg)
    litmodel = LitMLPInferModel(intergen_model, mlp_head, infer_cfg).to(torch.device("cuda:0"))

    # Load prompts
    with open("prompts.txt") as f:
        texts = f.readlines()
    texts = [text.strip("\n") for text in texts]

    # Create output directory
    timestamp = datetime.now().strftime("%m-%d-%y_%H-%M")
    result_dir = os.path.join("results", "mlp_outputs", timestamp)
    os.makedirs(result_dir, exist_ok=True)

    # Generate outputs
    for text in texts:
        name = text[:48]
        for i in range(3):
            litmodel.generate_one_sample(text, name + f"_{i}")