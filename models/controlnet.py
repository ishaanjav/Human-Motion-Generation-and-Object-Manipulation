import torch
import torch.nn as nn
from .utils import zero_module
from .intergen import InterGen

class ControlNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # Import InterDenoiser here to avoid circular import
        from .nets import InterDenoiser
        
        # Original network
        self.main_net = InterGen(cfg)  # Use full InterGen model
        
        # Load pre-trained weights for main network
        if cfg.CHECKPOINT:
            ckpt = torch.load(cfg.CHECKPOINT, map_location="cpu")
            # Handle the state dict keys properly
            state_dict = {}
            for k, v in ckpt["state_dict"].items():
                if "model" in k:
                    state_dict[k.replace("model.", "")] = v
            self.main_net.load_state_dict(state_dict, strict=False)
            print("Pre-trained weights loaded for main network!")
        
        # Freeze main network
        for param in self.main_net.parameters():
            param.requires_grad = False
        
        # Control network (copy of main network)
        self.control_net = InterDenoiser(
            cfg.INPUT_DIM, 
            cfg.LATENT_DIM,
            ff_size=cfg.FF_SIZE,
            num_layers=cfg.NUM_LAYERS,
            num_heads=cfg.NUM_HEADS,
            dropout=cfg.DROPOUT,
            activation=cfg.ACTIVATION,
            cfg_weight=cfg.CFG_WEIGHT
        )
        
        # Optionally freeze control network blocks
        if cfg.FREEZE_CONTROL_BLOCKS:
            for param in self.control_net.blocks.parameters():
                param.requires_grad = False
        
        # Zero convolution for initial control signal
        self.zero_conv_input = zero_module(nn.Conv1d(cfg.CONTROL_DIM, cfg.LATENT_DIM, 1))
        
        # Zero convolutions for each block's output
        self.zero_convs = nn.ModuleList([
            zero_module(nn.Conv1d(cfg.LATENT_DIM, cfg.LATENT_DIM, 1))
            for _ in range(cfg.NUM_LAYERS)
        ])
        
    def forward(self, x, timesteps, control, mask=None, cond=None):
        """
        x: B, T, D - motion features
        timesteps: B - diffusion timesteps
        control: B, T, C - control signal (e.g. trajectory, pose, etc)
        mask: B, T - sequence mask
        cond: B, D - text condition
        """
        # Original network output: F(x; Θ)
        main_out = self.main_net.forward_test({"motion": x, "timesteps": timesteps, "mask": mask, "text": cond})["output"]
        
        # Initial control signal processing: Z(c; Θ_z1)
        control_emb = self.zero_conv_input(control.transpose(1, 2)).transpose(1, 2)
        
        # Add control signal to input: x + Z(c; Θ_z1)
        x_control = x + control_emb
        
        # Process through control network blocks with control injection
        h = x_control
        control_outputs = []
        
        # Process through each block with control injection
        for i, (block, zero_conv) in enumerate(zip(self.control_net.blocks, self.zero_convs)):
            # Process through block
            h = block(h, timesteps, mask, cond)
            
            # Add control signal through zero convolution
            h = h + zero_conv(control_emb.transpose(1, 2)).transpose(1, 2)
            
            # Store intermediate output
            control_outputs.append(h)
        
        # Final output: F(x; Θ) + Z(F(...); Θ_z2)
        return main_out + control_outputs[-1]