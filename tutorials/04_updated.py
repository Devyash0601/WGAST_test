# === Imports ===
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import time

import sys
import os
# Add the project root to sys.path to allow imports from other folders
sys.path.append(os.path.abspath('..'))

from runner.experiment import Experiment as OriginalExperiment # Rename to avoid conflict

# === Modified Experiment Class ===
class ResumableExperiment(OriginalExperiment):
    
    def train(self, train_dir, patch_size, patch_stride, batch_size, num_workers, epochs):
        """
        Trains the WGAST model with checkpointing for resumable training.
        """
        start_epoch = 0
        checkpoint_path = Path(self.opt.save_dir) / 'checkpoint.pth'

        # Check for and load a previous checkpoint
        if checkpoint_path.exists():
            print("Checkpoint found. Resuming training...")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"Resuming from epoch {start_epoch}.")
        else:
            print("No checkpoint found. Starting training from scratch.")

        # Your original training code goes here, but the loop needs to be adjusted
        for epoch in range(start_epoch, epochs):
            # ... The rest of your training loop for a single epoch goes here ...
            
            # Placeholder for the actual training logic
            # For demonstration, let's assume a simple loop
            print(f"Training epoch {epoch+1}/{epochs}")
            
            # --- The actual training step for a single epoch ---
            # self.model.train()
            # ...
            
            # Save a checkpoint at the end of each epoch
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, checkpoint_path)
            print(f"Checkpoint saved for epoch {epoch + 1}.")

        print("Training completed.")
        return self.model # Or whatever your original method returned

# === Configuration Class ===
class Options:
    # ... (Your existing Options class) ...
    def __init__(self):
        self.lr = 2e-4
        self.batch_size = 32
        self.epochs = 110
        self.cuda = True
        self.ngpu = 1
        self.num_workers = 8
        self.save_dir = Path('../data/Tdivision')
        self.data_dir = Path('../data/Tdivision')
        self.train_dir = Path('../data/Tdivision/train')
        self.test_dir = Path('../data/Tdivision/test')
        self.image_size = [400, 400]
        self.patch_size = [32, 32]
        self.patch_stride = 8
        self.test_patch = 32
        self.ifAdaIN = True
        self.ifAttention = True
        self.ifTwoInput = False
        self.a = 1e-2
        self.b = 1
        self.c = 1
        self.d = 1

# === Main execution block ===
opt = Options()

# Set up CUDA and reproducibility
torch.manual_seed(2024)
opt.cuda = torch.cuda.is_available()
if opt.cuda:
    torch.cuda.manual_seed_all(2024)
    cudnn.benchmark = True
    cudnn.deterministic = True

# Use the modified ResumableExperiment class
experiment = ResumableExperiment(opt)

# --- Training and Testing ---
if opt.epochs > 0:
    start_time = time.time()
    
    # Run the training process
    predictions = experiment.train(opt.train_dir,
                                   opt.patch_size, 
                                   opt.patch_stride, 
                                   opt.batch_size,
                                   num_workers=1, 
                                   epochs=opt.epochs)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training took {elapsed_time:.2f} seconds")

# Test the model
results = experiment.test(opt.test_dir,
                          opt.patch_size,
                          num_workers=1)

print("Testing completed.")