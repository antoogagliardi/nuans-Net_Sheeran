import os
import torch
import pandas as pd
from pprint import pprint
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from transformers import BertTokenizer

import wandb

from src.utils import read_config_file, construct_data, slice_txt_data, open_data, save_data
from src.utils import sample_data
from src.data import LyricDataset, DataManager
from src.model import BERT_SongWriter

# Folder creation
os.makedirs("ckpt", exist_ok=True)
os.makedirs("wandb", exist_ok=True)

# Reading configuration file
cfg = read_config_file(config_path="./configs/config.yaml")
pprint(cfg)

# Paths setup
root_folder = "../"
print("Project Root Folder: ", root_folder)
cwd = os.getcwd()
print("Current work directory: ", cwd)
data_path = os.path.join(root_folder, cfg["paths"]["data"])
print("data path: ", data_path)
model_folder = os.path.join(root_folder, cfg["paths"]["model_files"])
print("model files folder: ", model_folder)
dataset_path = os.path.join(data_path, "raw/dataset_header_fullEnglish.csv")
print("dataset path: ", dataset_path)
ckpt_path = os.path.join(cwd, cfg["training"]["ckpt_path"])
print("ckpt path: ", ckpt_path)

# Device setup
device = torch.device(cfg["project"]["device"])
print(f"device used: {device}")



# Data reading
print("Length Dataset: ", len(pd.read_csv(dataset_path)))
pprint(pd.read_csv(dataset_path).head())
txt_lines, txt_lines_pairs = construct_data(data_file=dataset_path)
print("\n")
print("--- Recap Files Created ---")
print(f" -- Txt Lines: {type(txt_lines)}")
print(f"   |- Length Txt Lines: {len(txt_lines)}")
print(f" -- Txt Lines Pairs: {type(txt_lines_pairs)}")
print(f"   |- Txt Lines Pairs Elem: {type(txt_lines_pairs[0])}")
print(f"   |- Length Txt Lines Pairs: {len(txt_lines_pairs)}")

save_data(txt_lines, type="lines", path=os.path.join(data_path, "raw/txt_lines.txt"))
save_data(txt_lines_pairs, type="lines-pairs", path=os.path.join(model_folder, "txt_lines_pairs.txt"))

train_txt_lines, valid_txt_lines, test_txt_lines = slice_txt_data(txt_lines, n_slice=3)
save_data(train_txt_lines, type="lines", path=os.path.join(data_path, "train_txt_lines.txt"))
save_data(valid_txt_lines, type="lines", path=os.path.join(data_path, "valid_txt_lines.txt"))
save_data(test_txt_lines, type="lines", path=os.path.join(data_path, "test_txt_lines.txt"))

train_txt_lines = open_data(path=os.path.join(data_path, "train_txt_lines.txt"), type="lines")
valid_txt_lines = open_data(path=os.path.join(data_path, "valid_txt_lines.txt"), type="lines")

gold_txt_lines_pairs = open_data(path=os.path.join(model_folder, "txt_lines_pairs.txt"), type="lines-pairs")

print(f"Length train_txt_lines : {len(train_txt_lines)}")
print(f"Length valid_txt_lines : {len(valid_txt_lines)}")
print("\n")
print(f"Length of gold text-lines pairs: {len(gold_txt_lines_pairs)}")

N_SAMPLES = 1000
N_ALTERNATIVES = 15
train_in_data, train_ground_truth = sample_data(txt_lines=train_txt_lines, txt_lines_pair=gold_txt_lines_pairs,
                                                n_samples=N_SAMPLES, n_alternatives=N_ALTERNATIVES)
valid_in_data, valid_ground_truth = sample_data(txt_lines=valid_txt_lines, txt_lines_pair=gold_txt_lines_pairs,
                                                n_samples=N_SAMPLES, n_alternatives=N_ALTERNATIVES)
pprint(f" -- Input Txt Lines Pairs: {type(train_in_data)} {type(valid_in_data)}")
print(f"   |- Length Input Txt Lines Pairs: {len(train_in_data)} {len(valid_in_data)}")
print(f" -- Ground Truth {type(train_ground_truth)} {type(valid_ground_truth)}")
print(f"   |- Length Ground Truth: {len(train_ground_truth)} {len(valid_ground_truth)}")


train_lyric_dataset = LyricDataset(in_data=train_in_data, gold_data=train_ground_truth,
                                   tokenizer=BertTokenizer.from_pretrained("bert-base-uncased"))
valid_lyric_dataset = LyricDataset(in_data=valid_in_data, gold_data=valid_ground_truth,
                                   tokenizer=BertTokenizer.from_pretrained("bert-base-uncased"))
print("Length of Training Dataset: ", len(train_lyric_dataset))
print("Length of Validation Dataset: ", len(valid_lyric_dataset))

BATCH_SIZE = 16
data_manager = DataManager(train_data=train_lyric_dataset, valid_data=valid_lyric_dataset)



# Model Training Loop
    # Prepare the training
config = cfg["training"]
ckpt_path = os.path.join(os.getcwd(), config["ckpt_path"])
print("ckpt path: ", ckpt_path)
PROJECT_NAME = config["wandb_proj"]
RUN_NAME = config["wandb_run"]
# LAST_RUN_ID is discovered once the project is created
EPOCHS = config["epochs"]
ACCUMULATE_BATCH = config["grad_accumulation"]

    # Model Instantiation
model = BERT_SongWriter(device=device)
if device.type == "mps":    # Note: MPS Backend doesn't support torch.DoubleTensor(=float64)
    model = model.to(device=device, dtype=torch.float32)
else:
    model = model.to(device=device)
pprint(model)

    # Training loop
RESUME_TRAIN = config["resume"]
if RESUME_TRAIN == True:
    LAST_RUN_ID = config["wandb_runID"]            # LAST_RUN_ID is discovered once the project is created
    LAST_EPOCH = config["last_epoch"]
    ADD_EPOCHS = config["add_epoch"]
    run = wandb.init(project=PROJECT_NAME, name=RUN_NAME,
                     config=config,
                     resume=True, id=LAST_RUN_ID)
    ckpt = os.path.join(ckpt_path, f"{LAST_RUN_ID}/model-epoch={LAST_EPOCH-1}.ckpt")
    EPOCHS = LAST_EPOCH + ADD_EPOCHS
else:
    run = wandb.init(project=PROJECT_NAME,
                     name=RUN_NAME,config=config)


wandb_logger = WandbLogger(name=RUN_NAME,
                           save_dir=ckpt_path,
                           offline=False,
                           project=PROJECT_NAME, log_model=False)   # log_model = "all"/True/False

ckpt_callback = ModelCheckpoint(dirpath=f"{ckpt_path}/{wandb.run.id}",
                                filename='{epoch}', save_top_k=-1, every_n_epochs=1)
es_callback = EarlyStopping(monitor="valid.loss", min_delta=0.001, patience=5, mode="min")

trainer = pl.Trainer(accelerator=device.type,                       # gpu, cpu, mps
                     num_sanity_val_steps=1,
                     logger=wandb_logger,
                     devices=1, max_epochs=EPOCHS,
                     callbacks=[ckpt_callback, es_callback],
                     log_every_n_steps=1)
trainer.wandb_id    = wandb.run.id
trainer.device      = device
trainer.batch_size  = BATCH_SIZE

training_dataloader = data_manager.train_dataloader(batch_size=BATCH_SIZE, drop_last=False, suffle_data=False)
validation_dataloader = data_manager.val_dataloader(batch_size=BATCH_SIZE, drop_last=False)

# Start the training procedure
if RESUME_TRAIN == True:
    trainer.fit(model,
                train_dataloaders=training_dataloader, val_dataloaders=validation_dataloader,
                ckpt_path=ckpt)
else:
    trainer.fit(model,
                train_dataloaders=training_dataloader, val_dataloaders=validation_dataloader)
wandb.finish()
print("Training Complete")

