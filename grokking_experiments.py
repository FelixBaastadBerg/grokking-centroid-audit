import random
import time
import torch
import torch.nn as nn
import json
import os
from logger import MetricsLogger
from torch.utils.data import DataLoader
from utils import (evaluate, 
                   softmax_cross_entropy,
                   get_specified_args,
                   get_dataset,
                   get_model,
                   parse_args,
                   get_optimizer,
                   stablemax_cross_entropy)

# Helper function to make irregular log epochs
def make_irregular_log_epochs(num_epochs: int) -> set[int]:
    log_epochs = set()

    # 0..500 every 50
    for e in range(0, min(num_epochs, 500) + 1, 50):
        log_epochs.add(e)

    # 500..5000 every 500
    start = 500
    end = min(num_epochs, 5000)
    for e in range(start, end + 1, 500):
        log_epochs.add(e)

    # 5000..num_epochs every 5000
    start = 5000
    if num_epochs >= start:
        for e in range(start, num_epochs + 1, 5000):
            log_epochs.add(e)

    # Always log beginning and end
    log_epochs.add(0)
    log_epochs.add(num_epochs)

    return log_epochs


torch.set_num_threads(5) 

parser, args = parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)

train_dtype = getattr(torch, args.train_dtype)

device = args.device
print("Using device:", device)

train_dataset, test_dataset = get_dataset(args)
if args.full_batch:
    args.batch_size = len(train_dataset)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

torch.save(train_dataset, "last_train_loader.pt")
torch.save(test_dataset, "last_test_loader.pt")

args.lr = args.lr/(args.alpha**2)

model = get_model(args)
logger = MetricsLogger(args.num_epochs, args.log_frequency, seed=args.seed)
optimizer = get_optimizer(model, args)

print(args.loss_function)

loss_functions = {
    "cross_entropy": softmax_cross_entropy,
    "stablemax": stablemax_cross_entropy
}
loss_function = loss_functions[args.loss_function]
ce_dtype = getattr(torch, args.cross_entropy_dtype)
save_model_checkpoints = range(0, args.num_epochs, args.log_frequency)
saved_models = {epoch: None for epoch in save_model_checkpoints}

softmax_temperature = 1

if args.full_batch:
    all_data = train_dataset.dataset.data[train_dataset.indices].to(device)
    all_targets = train_dataset.dataset.targets[train_dataset.indices].to(device).long()

    all_test_data = test_dataset.dataset.data[test_dataset.indices].to(device)
    all_test_targets = test_dataset.dataset.targets[test_dataset.indices].to(device).long()

    if not (args.use_transformer or args.use_embedding):
        all_data = all_data.to(train_dtype)
        all_test_data = all_test_data.to(train_dtype)
else:
    raise ValueError("Current implementation only supports full batch training.")

loss = torch.inf
start_time = time.time()
model.to(device).to(train_dtype)
log_epochs = make_irregular_log_epochs(args.num_epochs)  # since your loop goes 0..num_epochs-1

for epoch in range(args.num_epochs+1):
    #Shuffling the data should not matter for full batch GD, 
    #but it sometimes does matter because of floating point errors
    permutation = torch.randperm(all_data.size(0))
    shuffled_data = all_data[permutation]
    shuffled_targets = all_targets[permutation]
    model.train()
    optimizer.zero_grad()
    output = model(shuffled_data) 
    if args.use_transformer:
        output = output[:, -1]
    output = output*args.alpha
    loss = loss_function(output, shuffled_targets, dtype=ce_dtype)
    loss.backward()
    optimizer.step()

    if epoch % logger.log_frequency == 0:
    #if epoch in log_epochs:
        logger.log_metrics(
            model=model,
            epoch=epoch,
            save_model_checkpoints=save_model_checkpoints,
            saved_models=saved_models,
            all_data=shuffled_data,
            all_targets=shuffled_targets,
            all_test_data=all_test_data,
            all_test_targets=all_test_targets,
            args=args,
            loss_function=loss_function,
        )

        print(f'Epoch {epoch}: Training loss: {loss.item():.4f}')
        if epoch > 0:
            print(f"Time taken for the last {args.log_frequency} epochs: {(time.time() - start_time)/60:.2f} min")
        start_time = time.time()

model.eval().to('cpu')
test_loss, test_accuracy = evaluate(model, test_loader, use_embedding=args.use_transformer or args.use_embedding)
print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}')
args.lr = args.lr

specified_args = get_specified_args(parser, args)

# --- build decision D1 -------------------------------------------------------
# The old key was '|'-joined, which is ILLEGAL in NTFS paths, so runs could not be
# written on Windows at all. It also only included args that differed from their
# defaults, so an MLP run and a Transformer run could collide and overwrite each
# other. arch/task/seed are now always present, and the separator is safe.
ARCH = "transformer" if args.use_transformer else "mlp"
_OP_TO_TASK = {"add_mod": "add", "product_mod": "mul",
               "subtract_mod": "sub", "divide_mod": "div"}
TASK = "parity" if args.dataset == "sparse_parity" else _OP_TO_TASK.get(
    args.binary_operation, args.binary_operation)

_parts = [f"arch-{ARCH}", f"task-{TASK}", f"seed-{args.seed}"]
_parts += [f'{k}-{specified_args[k]}' for k in specified_args
           if k not in ("use_transformer", "binary_operation", "dataset", "seed")]
experiment_key = "__".join(_parts)[:200]

torch.save(saved_models, 'last_run_saved_model_checkpoints.pt')
torch.save(optimizer, 'last_optimizer.pt')

os.makedirs(f"loggs/{experiment_key}", exist_ok=True)
logger.metrics_df.to_csv(f"loggs/{experiment_key}/metrics.csv", index=False)
logger.dims_df.to_csv(f"loggs/{experiment_key}/dimensions.csv", index=False)


with open(f"loggs/{experiment_key}/args.json", 'w') as f:
    json.dump(vars(args), f, indent=4)

# D1: explicit run identity. Analysis reads THIS, never the directory name.
run_meta = {
    "arch": ARCH,
    "task": TASK,
    "p": args.modulo,
    "lr": args.lr,
    "weight_decay": args.weight_decay,
    "train_fraction": args.train_fraction,
    "seed": args.seed,
    "num_epochs": args.num_epochs,
    "log_frequency": args.log_frequency,
    "loss_function": args.loss_function,
    "optimizer": args.optimizer,
    "batch_size": args.batch_size,
    "hidden_sizes": args.hidden_sizes,
    "beta2": args.beta2,
    "alpha": args.alpha,
    "cross_entropy_dtype": args.cross_entropy_dtype,
    "_source": "native",
}
with open(f"loggs/{experiment_key}/run_meta.json", 'w') as f:
    json.dump(run_meta, f, indent=2)

print(f"Saving run: {experiment_key}")