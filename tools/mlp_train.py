import sys
sys.path.append(sys.path[0] + r"/../")
import torch
import lightning.pytorch as pl
import torch.optim as optim
from collections import OrderedDict
from datasets import DataModule
from configs import get_config
from os.path import join as pjoin
from torch.utils.tensorboard import SummaryWriter
from models import *
from models.mlp_head import MLPHead

class LitMLPModel(pl.LightningModule):
    def __init__(self, intergen_model, mlp_head, cfg):
        super().__init__()
        self.cfg = cfg
        self.intergen_model = intergen_model
        self.mlp_head = mlp_head
        
        # Freeze InterGen model
        for param in self.intergen_model.parameters():
            param.requires_grad = False
            
        self.writer = SummaryWriter(pjoin(cfg.GENERAL.CHECKPOINT, cfg.GENERAL.EXP_NAME, 'log'))

    def _configure_optim(self):
        optimizer = optim.AdamW(self.mlp_head.parameters(), 
                              lr=float(self.cfg.TRAIN.LR), 
                              weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        scheduler = CosineWarmupScheduler(optimizer=optimizer, 
                                        warmup=10, 
                                        max_iters=self.cfg.TRAIN.EPOCH, 
                                        verbose=True)
        return [optimizer], [scheduler]

    def configure_optimizers(self):
        return self._configure_optim()

    def forward(self, batch_data):
        # Get motion outputs from InterGen
        with torch.no_grad():
            intergen_output = self.intergen_model.forward_test(batch_data)
            motion_output = intergen_output["output"]  # Shape: B, T, nfeats*2
            
        # Split InterGen output into first and last 256 dimensions
        first_half = motion_output[..., :256]  # First 256 dimensions
        second_half = motion_output[..., 256:]  # Last 256 dimensions
        
        # Process first half through MLP
        mlp_output = self.mlp_head(first_half)
        
        # Concatenate MLP output with untouched second half
        final_output = torch.cat([mlp_output, second_half], dim=-1)
        
        return final_output

    def training_step(self, batch, batch_idx):
        # Get MLP output (which includes both processed first half and untouched second half)
        mlp_output = self.forward(batch)
        
        # Split both the output and target into first and second halves
        output_first_half = mlp_output[..., :256]  # MLP processed part
        output_second_half = mlp_output[..., 256:]  # Untouched part
        
        target_first_half = batch["target"][..., :256]  # Target for first half
        target_second_half = batch["target"][..., 256:]  # Target for second half
        
        # Calculate loss only on the first half (MLP processed part)
        loss = torch.nn.MSELoss()(output_first_half, target_first_half)
        
        # Verify that second half remains unchanged
        assert torch.allclose(output_second_half, batch["target"][..., 256:]), "Second half should remain unchanged"
        
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(self.mlp_head.parameters(), 0.5)
        opt.step()

        return {"loss": loss}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.global_rank == 0:
            self.writer.add_scalar('train_loss', outputs['loss'], self.global_step)

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
    train_cfg = get_config("configs/train.yaml")
    data_cfg = get_config("configs/datasets.yaml").your_dataset  # You'll need to create this

    # Build models
    intergen_model, mlp_head = build_models(model_cfg)
    
    # Create Lightning model
    litmodel = LitMLPModel(intergen_model, mlp_head, train_cfg)
    
    # Create datamodule for your new dataset
    datamodule = DataModule(data_cfg, train_cfg.TRAIN.BATCH_SIZE, train_cfg.TRAIN.NUM_WORKERS)
    
    # Setup trainer
    trainer = pl.Trainer(
        default_root_dir=pjoin(train_cfg.GENERAL.CHECKPOINT, train_cfg.GENERAL.EXP_NAME),
        devices="auto", 
        accelerator='gpu',
        max_epochs=train_cfg.TRAIN.EPOCH,
        strategy=DDPStrategy(find_unused_parameters=True),
        precision=32,
    )

    trainer.fit(model=litmodel, datamodule=datamodule)