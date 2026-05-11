import sys, os, importlib

# Remove src from path to avoid shadowing torch
sys.path = [p for p in sys.path if 'muKG_LB' not in p]

# Add only the project root (not src) to path
sys.path.insert(0, '/home/hma/muKG_LB')

# Import torch first
import torch
import torch.nn as nn
print(f"torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")

# Now add src to path for project imports
sys.path.insert(0, '/home/hma/muKG_LB/src')

# Use importlib to load project modules
import importlib.util

# Load py.args_handler
spec = importlib.util.spec_from_file_location("py.args_handler", 
    "/home/hma/muKG_LB/src/py/args_handler.py")
py_args_handler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(py_args_handler)
load_args = py_args_handler.load_args

# Load py.load.kgs
spec = importlib.util.spec_from_file_location("py.load.kgs",
    "/home/hma/muKG_LB/src/py/load/kgs.py")
py_load_kgs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(py_load_kgs)
read_kgs_from_folder = py_load_kgs.read_kgs_from_folder

# Load py.base.losses
spec = importlib.util.spec_from_file_location("py.base.losses",
    "/home/hma/muKG_LB/src/py/base/losses.py")
py_base_losses = importlib.util.module_from_spec(spec)
spec.loader.exec_module(py_base_losses)
get_loss_func_torch = py_base_losses.get_loss_func_torch

# Load py.base.optimizers
spec = importlib.util.spec_from_file_location("py.base.optimizers",
    "/home/hma/muKG_LB/src/py/base/optimizers.py")
py_base_optimizers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(py_base_optimizers)
get_optimizer_torch = py_base_optimizers.get_optimizer_torch

# Load torch.kge_models.basic_model first (dependency)
spec = importlib.util.spec_from_file_location("torch.kge_models.basic_model",
    "/home/hma/muKG_LB/src/torch/kge_models/basic_model.py")
torch_kge_basic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(torch_kge_basic)

# Load torch.kge_models.TransE
spec = importlib.util.spec_from_file_location("torch.kge_models.TransE",
    "/home/hma/muKG_LB/src/torch/kge_models/TransE.py")
torch_kge_transe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(torch_kge_transe)
TransE = torch_kge_transe.TransE

# Load torch.kge_models.pytorch_dataloader
spec = importlib.util.spec_from_file_location("torch.kge_models.pytorch_dataloader",
    "/home/hma/muKG_LB/src/torch/kge_models/pytorch_dataloader.py")
torch_kge_dataloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(torch_kge_dataloader)
PyTorchTrainDataset = torch_kge_dataloader.PyTorchTrainDataset

from torch.utils.data import DataLoader
import numpy as np

# Load args
model_name = 'transe'
args = load_args('src/py/experiments/args_kge/' + model_name + r'_fb15k237_args.json')
args.training_data = 'src/py/data/FB15K237/'
args.max_epoch = 1

print(f"batch_size: {args.batch_size}")
print(f"neg_triple_num: {args.neg_triple_num}")

# Load data
kgs = read_kgs_from_folder('lp', args.training_data, args.dataset_division, 
                            args.alignment_module, args.ordered, remove_unlinked=False)

# Setup model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = TransE(kgs, args)
model.to(device)

optimizer = get_optimizer_torch(args.optimizer, model, args.learning_rate)

# Setup dataloader
train_dataset = PyTorchTrainDataset(kgs.relation_triples_list, args.neg_triple_num, kgs)
data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                         collate_fn=train_dataset.collate_fn,
                         shuffle=True, pin_memory=True, num_workers=args.batch_threads_num,
                         drop_last=False)

# Run ~20 steps for profiling
print("Starting training loop (20 steps)...")
model.train()
step_count = 0
for step, data_raw in enumerate(data_loader):
    if step >= 20:
        break
    
    optimizer.zero_grad()
    batch_size = int(data_raw[0].shape[0] / (args.neg_triple_num + 1))
    
    data = {
        'batch_h': data_raw[0].to(device),
        'batch_r': data_raw[1].to(device),
        'batch_t': data_raw[2].to(device)
    }
    
    score = model(data)
    
    po_score = score[:batch_size].view(batch_size, -1)
    ne_score = score[batch_size:].view(batch_size, -1)
    loss = get_loss_func_torch(po_score, ne_score, args)
    
    loss.backward()
    optimizer.step()
    
    step_count += 1
    if step_count % 5 == 0:
        print(f"Step {step}, loss: {loss.item():.4f}")

print(f"Training complete! Ran {step_count} steps.")
