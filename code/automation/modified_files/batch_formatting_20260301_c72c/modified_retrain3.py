# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import time
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import datasets
import utils
from model import CNN
from nni.nas.pytorch.utils import AverageMeter
from nni.retiarii import fixed_arch

logger = logging.getLogger('nni')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()


def execution_flow(config, train_data, model, opt, loss_func, epoch):
    accuracy_top1 = AverageMeter("top1")
    accuracy_top5 = AverageMeter("top5")
    loss_values = AverageMeter("losses")

    current_step = epoch * len(train_data)
    learning_rate = opt.param_groups[0]["lr"]
    logger.info("Epoch %d LR %.6f", epoch, learning_rate)
    writer.add_scalar("lr", learning_rate, global_step=current_step)

    model.train()

    for iteration, (input_data, target) in enumerate(train_data):
        input_data, target = input_data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        batch_size = input_data.size(0)

        opt.zero_grad()
        predictions, aux_predictions = model(input_data)
        total_loss = loss_func(predictions, target)
        if config.aux_weight > 0.:
            total_loss += config.aux_weight * loss_func(aux_predictions, target)
        total_loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        opt.step()

        metrics = utils.accuracy(predictions, target, topk=(1, 5))
        loss_values.update(total_loss.item(), batch_size)
        accuracy_top1.update(metrics["acc1"], batch_size)
        accuracy_top5.update(metrics["acc5"], batch_size)
        writer.add_scalar("loss/train", total_loss.item(), global_step=current_step)
        writer.add_scalar("acc1/train", metrics["acc1"], global_step=current_step)
        writer.add_scalar("acc5/train", metrics["acc5"], global_step=current_step)

        if iteration % config.log_frequency == 0 or iteration == len(train_data) - 1:
            logger.info(
                "Train: [{:3d}/{}] Step {:03d}/{:03d} Loss {loss_values.avg:.3f} "
                "Prec@(1,5) ({accuracy_top1.avg:.1%}, {accuracy_top5.avg:.1%})".format(
                    epoch + 1, config.epochs, iteration, len(train_data) - 1, loss_values=loss_values,
                    accuracy_top1=accuracy_top1, accuracy_top5=accuracy_top5))

        current_step += 1

    logger.info("Train: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, config.epochs, accuracy_top1.avg))


def evaluate(config, valid_data, model, loss_func, epoch, current_step):
    accuracy_top1 = AverageMeter("top1")
    accuracy_top5 = AverageMeter("top5")
    loss_values = AverageMeter("losses")

    model.eval()

    with torch.no_grad():
        for iteration, (input_data, target) in enumerate(valid_data):
            input_data, target = input_data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            batch_size = input_data.size(0)

            predictions = model(input_data)
            total_loss = loss_func(predictions, target)

            metrics = utils.accuracy(predictions, target, topk=(1, 5))
            loss_values.update(total_loss.item(), batch_size)
            accuracy_top1.update(metrics["acc1"], batch_size)
            accuracy_top5.update(metrics["acc5"], batch_size)

            if iteration % config.log_frequency == 0 or iteration == len(valid_data) - 1:
                logger.info(
                    "Valid: [{:3d}/{}] Step {:03d}/{:03d} Loss {loss_values.avg:.3f} "
                    "Prec@(1,5) ({accuracy_top1.avg:.1%}, {accuracy_top5.avg:.1%})".format(
                        epoch + 1, config.epochs, iteration, len(valid_data) - 1, loss_values=loss_values,
                        accuracy_top1=accuracy_top1, accuracy_top5=accuracy_top5))

    writer.add_scalar("loss/test", loss_values.avg, global_step=current_step)
    writer.add_scalar("acc1/test", accuracy_top1.avg, global_step=current_step)
    writer.add_scalar("acc5/test", accuracy_top5.avg, global_step=current_step)

    logger.info("Valid: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, config.epochs, accuracy_top1.avg))

    return accuracy_top1.avg


if __name__ == "__main__":
    parser = ArgumentParser("darts")
    parser.add_argument("--layers", default=20, type=int)
    parser.add_argument("--batch-size", default=96, type=int)
    parser.add_argument("--log-frequency", default=10, type=int)
    parser.add_argument("--epochs", default=600, type=int)
    parser.add_argument("--aux-weight", default=0.4, type=float)
    parser.add_argument("--drop-path-prob", default=0.2, type=float)
    parser.add_argument("--workers", default=4)
    parser.add_argument("--grad-clip", default=5., type=float)
    parser.add_argument("--arc-checkpoint", default="./checkpoints/epoch_0.json")

    arguments = parser.parse_args()
    dataset_train, dataset_valid = datasets.get_dataset("cifar10", cutout_length=16)

    with fixed_arch(arguments.arc_checkpoint):
        network = CNN(32, 3, 36, 10, arguments.layers, auxiliary=True)
    criterion = nn.CrossEntropyLoss()

    network.to(device)
    criterion.to(device)

    optimizer = torch.optim.SGD(network.parameters(), 0.025, momentum=0.9, weight_decay=3.0E-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, arguments.epochs, eta_min=1E-6)

    train_loader = torch.utils.data.DataLoader(dataset_train,
                                               batch_size=arguments.batch_size,
                                               shuffle=True,
                                               num_workers=arguments.workers,
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(dataset_valid,
                                               batch_size=arguments.batch_size,
                                               shuffle=False,
                                               num_workers=arguments.workers,
                                               pin_memory=True)

    best_accuracy = 0.
    for epoch in range(arguments.epochs):
        drop_probability = arguments.drop_path_prob * epoch / arguments.epochs
        network.drop_path_prob(drop_probability)

        # training
        execution_flow(arguments, train_loader, network, optimizer, criterion, epoch)

        # validation
        current_step = (epoch + 1) * len(train_loader)
        top1 = evaluate(arguments, valid_loader, network, criterion, epoch, current_step)
        best_accuracy = max(best_accuracy, top1)

        lr_scheduler.step()

    logger.info("Final best Prec@1 = {:.4%}".format(best_accuracy))