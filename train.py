
import math
import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import multiprocessing
from os.path import join
from datetime import datetime
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
torch.backends.cudnn.benchmark= True  # Provides a speedup

import util
import test
import parser
import commons
import datasets_ws
import network
from model.sync_batchnorm import convert_model

from loss import tripletLossDTW
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
#### Initial setup: parser, logging...
args = parser.parse_arguments()
start_time = datetime.now()
args.save_dir = join("logs", args.save_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")
logging.info(f"Using {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs")


#### Creation of Datasets
logging.debug(f"Loading dataset {args.dataset_name} from folder {args.datasets_folder}")

triplets_ds = datasets_ws.TripletsDataset(args, args.datasets_folder, args.dataset_name, "train", args.negs_num_per_query)
logging.info(f"Train query set: {triplets_ds}")

val_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "val")
logging.info(f"Val set: {val_ds}")

test_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "test")
logging.info(f"Test set: {test_ds}")

#### Initialize model
model = network.GeoLocalizationNet(args)

model = model.to(args.device)

model = torch.nn.DataParallel(model)

### Setup Optimizer and Loss
if args.optim == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
elif args.optim == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001)

criterion_triplet = nn.TripletMarginLoss(margin=args.margin, p=2, reduction="sum")

#### Resume model, optimizer, and other training parameters
if args.resume:
    model, _, best_r1pr5, start_epoch_num, not_improved_num = util.resume_train(args, model, strict=False)
    logging.info(f"Resuming from epoch {start_epoch_num} with best recall@5 {best_r1pr5:.1f}")
else:
    best_r1pr5 = start_epoch_num = not_improved_num = 0

logging.info(f"Output dimension of the model is {args.features_dim}")

if torch.cuda.device_count() >= 2:
    # When using more than 1GPU, use sync_batchnorm for torch.nn.DataParallel
    model = convert_model(model)
    model = model.cuda()

align_triplet = tripletLossDTW().to(args.device)
#### Training loop
for epoch_num in range(start_epoch_num, args.epochs_num):
    logging.info(f"Start training epoch: {epoch_num:02d}")
    
    epoch_start_time = datetime.now()
    epoch_losses = np.zeros((0,1), dtype=np.float32)
    
    # How many loops should an epoch last (default is 5000/1000=5)
    loops_num = math.ceil(args.queries_per_epoch / args.cache_refresh_rate)
    for loop_num in range(loops_num):
        logging.debug(f"Cache: {loop_num} / {loops_num}")
        
        # Compute triplets to use in the triplet loss
        triplets_ds.is_inference = True
        triplets_ds.compute_triplets(args, model)
        triplets_ds.is_inference = False
        
        triplets_dl = DataLoader(dataset=triplets_ds, num_workers=args.num_workers,
                                 batch_size=args.train_batch_size,
                                 collate_fn=datasets_ws.collate_fn,
                                 pin_memory=(args.device=="cuda"),
                                 drop_last=True)
        
        model = model.train()
        # images shape: (train_batch_size*4)*3*H*W
        # triplets_local_indexes shape: (train_batch_size*2)*3 ; because 2 triplets per query
        for images, triplets_local_indexes, _ in tqdm(triplets_dl, ncols=100):
            
            # Flip all triplets or none
            if args.horizontal_flip:
                images = transforms.RandomHorizontalFlip()(images)
            
            # Compute features of all images (images contains queries, positives and negatives)
            
            features_a, features = model(images.to(args.device))
            loss_triplet = 0
            loss_triplet_global = 0
            loss_triplet_local = 0

            if args.criterion == "triplet":
                triplets_local_indexes = torch.transpose(
                    triplets_local_indexes.view(args.train_batch_size, args.negs_num_per_query, 3), 1, 0)
                for triplets in triplets_local_indexes:
                    queries_indexes, positives_indexes, negatives_indexes = triplets.T

                    loss_triplet_global += criterion_triplet(features[queries_indexes],
                                                      features[positives_indexes],
                                                      features[negatives_indexes])
                    loss_triplet_local += align_triplet([features_a[queries_indexes],
                                                      features_a[positives_indexes],
                                                      features_a[negatives_indexes]])
                       
            del features
            del features_a 
            loss_triplet_global /= (args.train_batch_size * args.negs_num_per_query)
            loss_triplet_local /= (args.train_batch_size * args.negs_num_per_query)
            loss_triplet = loss_triplet_global + loss_triplet_local

            optimizer.zero_grad()
            loss_triplet.backward()
            optimizer.step()

            batch_loss = loss_triplet.item()
            epoch_losses = np.append(epoch_losses, batch_loss)
            del loss_triplet
            # del loss_triplet_global
            # del loss_triplet_local
        
        logging.debug(f"global loss = {loss_triplet_global.item():.6f},  " +
                      f"local loss = {loss_triplet_local.item():.6f}")
        logging.debug(f"Epoch[{epoch_num:02d}]({loop_num}/{loops_num}): " +
                      f"Current batch triplet loss = {batch_loss:.4f}, " +
                      f"Average epoch triplet loss = {epoch_losses.mean():.4f}")
    
    logging.info(f"Finished epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                 f"Average epoch triplet loss = {epoch_losses.mean():.4f}")
    
    # Compute recalls on validation set
    recalls, recalls_str = test.test(args, val_ds, model)
    logging.info(f"Recalls on val set {val_ds}: {recalls_str}")
    
    is_best = recalls[0] + recalls[1] > best_r1pr5
    
    # Save checkpoint, which contains all training parameters
    util.save_checkpoint(args, {"epoch_num": epoch_num, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "recalls": recalls, "best_r1pr5": best_r1pr5,
        "not_improved_num": not_improved_num
    }, is_best, filename="last_model.pth")
    
    # If recall@5 did not improve for "many" epochs, stop training
    if is_best:
        logging.info(f"Improved: previous best R@1+R@5 = {best_r1pr5:.1f}, current R@1+R@5 = {(recalls[0]+recalls[1]):.1f}")
        best_r1pr5 = recalls[0] + recalls[1]
        not_improved_num = 0
    else:
        not_improved_num += 1
        logging.info(f"Not improved: {not_improved_num} / {args.patience}: best R@1+R@5 = {best_r1pr5:.1f}, current R@1+R@5 = {(recalls[0]+recalls[1]):.1f}")
        if not_improved_num >= args.patience:
            logging.info(f"Performance did not improve for {not_improved_num} epochs. Stop training.")
            break


logging.info(f"Best R@1+R@5: {best_r1pr5:.1f}")
logging.info(f"Trained for {epoch_num+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

#### Test best model on test set
best_model_state_dict = torch.load(join(args.save_dir, "best_model.pth"))["model_state_dict"]
model.load_state_dict(best_model_state_dict)

recalls, recalls_str = test.test(args, test_ds, model, test_method=args.test_method)
logging.info(f"Recalls on {test_ds}: {recalls_str}")

