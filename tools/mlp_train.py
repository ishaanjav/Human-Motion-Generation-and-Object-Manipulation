import sys
sys.path.append(sys.path[0] + r"/../")
import torch
import lightning.pytorch as pl
import torch.optim as optim
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
import numpy as np
from configs import get_config
from os.path import join as pjoin
from torch.utils.tensorboard import SummaryWriter
from models import *
from models.mlp_head import MLPHead
from datetime import datetime

class NPYDataset(Dataset):
    def __init__(self, npy_path):
        """Load data from the saved numpy file"""
        self.samples = np.load(npy_path, allow_pickle=True)
        print(f"Loaded {len(self.samples)} samples from {npy_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "noisy_pose": torch.from_numpy(sample["noisy_pose"].squeeze()),
            "clean_pose": torch.from_numpy(sample["clean_pose"].squeeze()),
            "object_bps": torch.from_numpy(sample["object_bps"].squeeze()),
            "object_motion": torch.from_numpy(sample["object_motion"].squeeze())
        }

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
        # Process the input data directly since we're using our custom dataset
        noisy_pose = batch_data["noisy_pose"]  # shape: (batch_size=32, 204)
        object_bps = batch_data["object_bps"]
        object_motion = batch_data["object_motion"]  # shape: (batch_size=32, 12)
        
        # Get motion outputs from InterGen
        with torch.no_grad():
            batch_size = noisy_pose.shape[0]  # Should be 32
            window_size = 210  # Same as in infer.py
            
            intergen_input = OrderedDict({
                "motion_lens": torch.ones(batch_size, 1).long().to(noisy_pose.device) * window_size,
                "text": ["The people carry a box"] * batch_size  # Same text for all samples in batch
            })
            
            intergen_output = self.intergen_model.forward_test(intergen_input)
            motion_output = intergen_output["output"]  # shape: (batch_size=32, seq_len=210, 524)
            print("InterGen output shape:", motion_output.shape)
            
        # Take first 262 dimensions from InterGen output
        first_half = motion_output[..., :262]  # (batch_size=32, seq_len=210, 262)
        
        # Concatenate first_half with object_motion for each timestep
        object_motion_expanded = object_motion.unsqueeze(1).expand(-1, window_size, -1)  # (batch_size=32, seq_len=210, 12)
        mlp_input = torch.cat([first_half, object_motion_expanded], dim=-1)  # (batch_size=32, seq_len=210, 274)
        
        # Process through MLP to get human pose
        # Reshape to (batch_size * seq_len, 274) for MLP
        mlp_input_reshaped = mlp_input.reshape(-1, 274)  # (batch_size*seq_len, 274)
        mlp_output = self.mlp_head(mlp_input_reshaped)  # (batch_size*seq_len, 204)
        mlp_output = mlp_output.reshape(batch_size, window_size, 204)  # (batch_size=32, seq_len=210, 204)
        
        return mlp_output

    def training_step(self, batch, batch_idx):
        # Get MLP output
        mlp_output = self.forward(batch)  # shape: (batch_size, seq_len, 204)
        
        # Get target from batch and expand it to match sequence length
        target = batch["clean_pose"]  # shape: (batch_size, 204)
        seq_len = mlp_output.shape[1]
        target = target.unsqueeze(1).expand(-1, seq_len, -1)  # shape: (batch_size, seq_len, 204)
        
        # Calculate loss between predicted pose and target pose
        loss = torch.nn.MSELoss()(mlp_output, target)
        
        # Log the loss
        self.log("train_loss", loss, prog_bar=True)
        
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        # Get MLP output
        mlp_output = self.forward(batch)  # shape: (batch_size, seq_len, 204)
        
        # Get target from batch and expand it to match sequence length
        target = batch["clean_pose"]  # shape: (batch_size, 204)
        seq_len = mlp_output.shape[1]
        target = target.unsqueeze(1).expand(-1, seq_len, -1)  # shape: (batch_size, seq_len, 204)
        
        # Calculate validation loss
        val_loss = torch.nn.MSELoss()(mlp_output, target)
        
        # Log the validation loss
        self.log("val_loss", val_loss, prog_bar=True)
        
        return {"val_loss": val_loss}

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
    
    # Create MLP head that takes 262 (InterGen) + 12 (object_motion) = 274 dimensions
    mlp_head = MLPHead(
        input_dim=274,  # 262 from InterGen + 12 from object_motion
        hidden_dims=[1024, 512, 512],
        output_dim=204  # Output human pose dimensions
    )
    
    return intergen_model, mlp_head

if __name__ == '__main__':
    model_cfg = get_config("configs/model.yaml")
    train_cfg = get_config("configs/train.yaml")
    
    # Build models
    intergen_model, mlp_head = build_models(model_cfg)
    
    # Create Lightning model
    litmodel = LitMLPModel(intergen_model, mlp_head, train_cfg)
    
    # Create dataset and dataloader
    npy_path = "/scratch/gpfs/ij9461/chois_release/samples/first_500_samples.npy"
    dataset = NPYDataset(npy_path)
    
    # Split dataset into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,  # Fixed batch size of 64
        shuffle=True,
        num_workers=64,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,  # Fixed batch size of 64
        shuffle=False,
        num_workers=64,
        pin_memory=True
    )
    
    EXP_NAME = f"MLP-{datetime.now().strftime('%H:%M')}"
    # Setup checkpointing
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=pjoin(train_cfg.GENERAL.CHECKPOINT, EXP_NAME, 'mlp_checkpoints'),
        filename='mlp-{epoch:02d}-{val_loss:.3f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min',
        save_last=True,
    )
    
    # Setup trainer with checkpointing
    trainer = pl.Trainer(
        default_root_dir=pjoin(train_cfg.GENERAL.CHECKPOINT, EXP_NAME),
        devices="auto", 
        accelerator='gpu',
        max_epochs=train_cfg.TRAIN.EPOCH,
        strategy="ddp",
        precision=32,
        callbacks=[checkpoint_callback],  # Add checkpoint callback
    )

    trainer.fit(model=litmodel, train_dataloaders=train_loader, val_dataloaders=val_loader)