import torch
from torch.nn.parallel import DistributedDataParallel

import copy 
import os
import sys
import logging

import numpy as np

from src.helper import (
    load_checkpoint,
    init_model,
    init_opt)

from src.datasets.text_dataset import make_text_dataset

from src.masks.utils import apply_masks
from src.masks.block import TextMaskCollator

from src.utils.logging import (
    CSVLogger,
    gpu_timer,
    grad_logger,
    AverageMeter)
from src.utils.tensors import repeat_interleave_batch



logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

def main(args):

    # modify later
    world_size = 1
    rank = 0

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder, predictor = init_model(device=device)
    target_encoder = copy.deepcopy(encoder)

    mask_collator = TextMaskCollator()


    log_file = os.path.join(args.logging_folder, f'{args.tag}.csv')
    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'loss'),
                           ('%.5f', 'mask-A'),
                           ('%.5f', 'mask-B'),
                           ('%d', 'time (ms)'))

    # Make data
    train_loader, val_loader, sampler = make_text_dataset(
        '../data/shakespeare/train.bin', 
        '../data/shakespeare/val.bin', 
        args.batch_size, 
        mask_collator, 
        training=True, 
        drop_last=True
        )
    
    for p in target_encoder.parameters():
        p.requires_grad = False

    # Load checkpoint if needed
    if args.load_checkpoint:
        encoder, predictor, optimizer = load_checkpoint(encoder=encoder,
                                                        predictor=predictor,
                                                        optimizer=optimizer,
                                                        path=args.checkpoint_path,)

    # Initialize optimizer
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
            encoder=encoder,
            predictor=predictor,
            wd=args.wd,
            final_wd=args.final_wd,
            start_lr=args.start_lr,
            ref_lr=args.lr,
            final_lr=args.final_lr,
            iterations_per_epoch=args.ipe,
            warmup=args.warmup,
            num_epochs=args.num_epochs,
            ipe_scale=args.ipe_scale,
            use_bfloat16=args.use_bfloat16)
    
    for p in target_encoder.parameters():
        p.requires_grad = False

    momentum_scheduler = (args.ema[0] + i*(args.ema[1]-args.ema[0])/(args.ipe*args.num_epochs*args.ipe_scale)
                          for i in range(int(args.ipe*args.num_epochs*args.ipe_scale)+1))

    def save_checkpoint(epoch):
        save_dict = {
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': args.batch_size,
            'world_size': world_size,
            'lr': args.lr
        }
        if rank == 0:
            torch.save(save_dict, args.checkpoint_path)
            if (epoch + 1) % args.checkpoint_freq == 0:
                torch.save(save_dict, args.checkpoint_path.format(epoch=f'{epoch + 1}'))


    for epoch in range(args.start_epoch, args.end_epoch):
        logger.info(f"Epoch {epoch+1}")

        loss_meter = AverageMeter()
        maskA_meter = AverageMeter()
        maskB_meter = AverageMeter()
        time_meter = AverageMeter()

        # Train loop
        for itr, (udata, masks_enc, masks_pred) in enumerate(train_loader):
            def load_texts():
                texts = udata[0].to(device)
                masks_1 = [u.to(device, non_blocking=True) for u in masks_enc]
                masks_2 = [u.to(device, non_blocking=True) for u in masks_pred]
                return (texts, masks_1, masks_2)
            
            texts, masks_enc, masks_pred = load_texts()
            maskA_meter.update(len(masks_enc[0][0]))
            maskB_meter.update(len(masks_pred[0][0]))

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

                def forward_target():
                    with torch.no_grad():
                        h = target_encoder(texts)
                        B = len(h)

                        h = apply_masks(h, masks_enc)
                        h = repeat_interleave_batch(h, B, repeat=len(masks_enc))

                def forward_context():
                    z = encoder(texts, masks_enc)
                    z = predictor(z, masks_enc, masks_pred)
                    return z
                
                def loss_fn(z, h):
                    loss = torch.nn.functional.l1_loss(z, h)
                    return loss
            
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=args.use_bfloat16):
                    h = forward_target()
                    z = forward_context()
                    loss = loss_fn(z, h)

                if args.use_bfloat16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                
                grad_stats = grad_logger(encoder.named_parameters())
                optimizer.zero_grad()

                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

                return (float(loss), _new_lr, _new_wd, grad_stats)
            (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            time_meter.update(etime)

            # -- Logging
            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, maskA_meter.val, maskB_meter.val, etime)
                if (itr % args.log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info('[%d, %5d] loss: %.3f '
                                'masks: %.1f %.1f '
                                '[wd: %.2e] [lr: %.2e] '
                                '[mem: %.2e] '
                                '(%.1f ms)'
                                % (epoch + 1, itr,
                                   loss_meter.avg,
                                   maskA_meter.avg,
                                   maskB_meter.avg,
                                   _new_wd,
                                   _new_lr,
                                   torch.cuda.max_memory_allocated() / 1024.**2,
                                   time_meter.avg))

                    if grad_stats is not None:
                        logger.info('[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
                                    % (epoch + 1, itr,
                                       grad_stats.first_layer,
                                       grad_stats.last_layer,
                                       grad_stats.min,
                                       grad_stats.max))

            log_stats()

            assert not np.isnan(loss), 'loss is nan'

        # -- Save Checkpoint after every epoch
        logger.info('avg. loss %.3f' % loss_meter.avg)
        save_checkpoint(epoch+1)


    # # Train loop
    # for epoch in range(args.num_epochs):
    #     for i, (input, target) in enumerate(train_loader):
    #         # Move to device
    #         input, target = input.to(device), target.to(device)

    #         # Forward
    #         encoded = encoder(input)
    #         pred = predictor(encoded)

    #         # Calculate loss
    #         loss = compute_loss(pred, target)

    #         # Backpropagate
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         if i % args.log_interval == 0:
    #             print(f"Epoch {epoch}, step {i}, loss {loss.item()}")

    #     # Validate
    #     val_loss = validate(val_loader, encoder, predictor)
    #     print(f"Epoch {epoch}, validation loss {val_loss}")

    #     # Save checkpoint
    #     save_checkpoint(encoder, predictor, optimizer, path=args.checkpoint_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--end_epoch', type=int, default=10)

    parser.add_argument('--load_checkpoint', action='store_true')
    parser.add_argument('--checkpoint_path', type=str, default='../checkpoints/shakespeare/pocket.pt')
    parser.add_argument('--checkpoint_freq', type=int, default=10)

    parser.add_argument('--lr', type=float, default=1.0e-3)
    parser.add_argument('--start_lr', type=float, default=2.0e-3)
    parser.add_argument('--final_lr', type=float, default=1.0e-6)
    
    parser.add_argument('--wd', type=float, default=0.04)
    parser.add_argument('--final_wd', type=float, default=0.4)
    
    parser.add_argument('--warmup', type=int, default=40)
    
    parser.add_argument('--ipe', type=int, default=100)
    parser.add_argument('--ipe_scale', type=float, default=1.0)

    parser.add_argument('--use_bfloat16', action='store_true')

    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--logging_folder', type=str, default='../logs/shakespeare')
    parser.add_argument('--tag', type=str, default='pocket')

    parser.add_argument('--ema', type=list, default=[0.996, 1.0])



    args = parser.parse_args()

    main(args)