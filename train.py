import argparse
import time

import torch.optim as optim
from torch.utils.data import DataLoader

from models import *
from datasets import *
from utils import *
import random
import test

hyp = {'k': 10.39,  # loss multiple
       'xy': 0.1367,  # xy loss fraction
       'wh': 0.01057,  # wh loss fraction
       'cls': 0.01181,  # cls loss fraction
       'conf': 0.8409,  # conf loss fraction
       'iou_t': 0.1287,  # iou target-anchor training threshold
       'lr0': 0.001028,  # initial learning rate
       'lrf': -3.441,  # final learning rate = lr0 * (10 ** lrf)
       'momentum': 0.9127,  # SGD momentum
       'weight_decay': 0.0004841,  # optimizer weight decay
       }


def train(
        cfg,
        data_cfg,
        img_size=416,
        epochs=273,  # 500200 batches at bs 64, dataset length 117263
        batch_size=16,
        accumulate=1,
        multi_scale=False,
        freeze_backbone=False,
):


    best_acc = -1
    init_seeds()
    weights = 'weights' + os.sep
    device = torch_utils.select_device()
    if multi_scale:
        img_size = 608  # initiate with maximum multi_scale size
        opt.num_workers = 0  # bug https://github.com/ultralytics/yolov3/issues/174
    else:
        torch.backends.cudnn.benchmark = True  # unsuitable for multiscale

    # Initialize model
    model = Darknet(cfg, img_size).to(device)

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=hyp['lr0'], momentum=hyp['momentum'], weight_decay=hyp['weight_decay'])

    start_epoch = 0

    cutoff = load_darknet_weights(model, weights + 'darknet53.conv.74')

    lf = lambda x: 1 - 10 ** (hyp['lrf'] * (1 - x / epochs))  # inv exp ramp to lr0 * 1e-2
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf, last_epoch=start_epoch - 1)

    dir_path = './data/label.txt'
    with open(dir_path, 'r') as file:
        img_files = file.read().splitlines()
    l = 4000
    img_files = img_files[0:l]
    random.shuffle(img_files)

    train_files = img_files[0:int(0.8*l)]
    val_files = img_files[int(0.8*l):l]

    # train_path = './data'
    dataset = LoadImagesAndLabels(train_files, img_size=img_size, augment=True)

    # Dataloader
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=opt.num_workers,
                            shuffle=True,
                            pin_memory=True,
                            collate_fn=dataset.collate_fn)

    # Mixed precision training https://github.com/NVIDIA/apex
    # install help: https://github.com/NVIDIA/apex/issues/259
    mixed_precision = False
    if mixed_precision:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    # Start training

    model.hyp = hyp  # attach hyperparameters to model
    model_info(model)
    nb = len(dataloader)
    results = (0, 0, 0, 0, 0)  # P, R, mAP, F1, test_loss
    n_burnin = min(round(nb / 5 + 1), 1000)  # burn-in batches
    os.remove('train_batch0.jpg') if os.path.exists('train_batch0.jpg') else None
    os.remove('test_batch0.jpg') if os.path.exists('test_batch0.jpg') else None
    for epoch in range(start_epoch, epochs):
        model.train()
        # print(('\n%8s%12s' + '%10s' * 7) % ('Epoch', 'Batch', 'xy', 'wh', 'conf', 'cls', 'total', 'nTargets', 'time'))

        # Update scheduler
        scheduler.step()

        # Freeze backbone at epoch 0, unfreeze at epoch 1
        if freeze_backbone and epoch < 2:
            for name, p in model.named_parameters():
                if int(name.split('.')[1]) < cutoff:  # if layer < 75
                    p.requires_grad = False if epoch == 0 else True

        mloss = torch.zeros(5).to(device)  # mean losses
        for i, (imgs, targets, _, _) in enumerate(dataloader):
            imgs = imgs.to(device)
            targets = targets.to(device)

            # Plot images with bounding boxes
            if epoch == 0 and i == 0:
                plot_images(imgs=imgs, targets=targets, fname='train_batch0.jpg')

            # SGD burn-in
            if epoch == 0 and i <= n_burnin:
                lr = hyp['lr0'] * (i / n_burnin) ** 4
                for x in optimizer.param_groups:
                    x['lr'] = lr

            # Run model
            pred = model(imgs)

            # Compute loss
            loss, loss_items = compute_loss(pred, targets, model)
            if torch.isnan(loss):
                print('WARNING: nan loss detected, ending training')
                return results

            # Compute gradient
            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Accumulate gradient for x batches before optimizing
            if (i + 1) % accumulate == 0 or (i + 1) == nb:
                optimizer.step()
                optimizer.zero_grad()

            mloss = (mloss * i + loss_items) / (i + 1)

            # Multi-Scale training (320 - 608 pixels) every 10 batches
            if multi_scale and (i + 1) % 10 == 0:
                dataset.img_size = random.choice(range(10, 20)) * 32
                print('multi_scale img_size = %g' % dataset.img_size)

        PATH = './model/model.pth'
        torch.save(model.state_dict(), PATH)
        with torch.no_grad():
            acc = test.test(val_files, cfg, img_size=img_size, model=model)

            if acc > best_acc:
                best_acc = acc
                PATH = './model/model_best.pth'
                torch.save(model.state_dict(), PATH)
            print(epoch, ': acc', acc, ' best:', best_acc)

    return results


def print_mutation(hyp, results):
    # Write mutation results
    a = '%11s' * len(hyp) % tuple(hyp.keys())  # hyperparam keys
    b = '%11.4g' * len(hyp) % tuple(hyp.values())  # hyperparam values
    c = '%11.3g' * len(results) % results  # results (P, R, mAP, F1, test_loss)
    print('\n%s\n%s\nEvolved fitness: %s\n' % (a, b, c))
    with open('evolve.txt', 'a') as f:
        f.write(c + b + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='size of each image batch')
    parser.add_argument('--accumulate', type=int, default=1, help='accumulate gradient x batches before optimizing')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='data/coco.data', help='coco.data file path')
    parser.add_argument('--multi-scale', action='store_true', help='random image sizes per batch 320 - 608')
    parser.add_argument('--img-size', type=int, default=416, help='pixels')
    parser.add_argument('--resume', action='store_true', help='resume training flag')
    parser.add_argument('--transfer', action='store_true', help='transfer learning flag')
    parser.add_argument('--num-workers', type=int, default=4, help='number of Pytorch DataLoader workers')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:9999', type=str, help='distributed training init method')
    parser.add_argument('--rank', default=0, type=int, help='distributed training node rank')
    parser.add_argument('--world-size', default=1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--nosave', action='store_true', help='do not save training results')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--evolve', action='store_true', help='run hyperparameter evolution')
    parser.add_argument('--var', default=0, type=int, help='debug variable')
    opt = parser.parse_args()
    print(opt, end='\n\n')

    if opt.evolve:
        opt.notest = True  # save time by only testing final epoch
        opt.nosave = True  # do not save checkpoints

    # Train
    results = train(
        opt.cfg,
        opt.data_cfg,
        img_size=opt.img_size,
        epochs=opt.epochs,
        batch_size=opt.batch_size,
        accumulate=opt.accumulate,
        multi_scale=opt.multi_scale,
    )



