from libauc.losses import AUCMLoss
from libauc.utils import ImbalancedDataGenerator
from libauc.sampler import DualSampler
from libauc.metrics import auc_roc_score

import torch 
import numpy as np
from optimizer import DPSGDA
from torchvision.datasets import MNIST
import math
import argparse
from utils import weights_init, set_all_seeds, ImageDataset, NeuralNetwork



def train(args):
    epsilon, delta = args.epsilon, args.delta
    clip_w, clip_v = args.clip_w, args.clip_v
    lr_w, lr_v = args.lr_w, args.lr_v
    imratio = args.imratio
    seed = args.seed
    batch_size = args.batch_size
    total_epochs = args.total_epochs
    set_all_seeds(seed)

    # NOTE: load data as numpy arrays
    d = MNIST(root="./data", train=True, download=True)
    train_data, train_targets = d.data, d.targets
    d = MNIST(root="./data", train=False)
    test_data, test_targets = d.data, d.targets

    # NOTE: get dataset size
    if imratio == 0.5: n=50000
    else: n=33995

    # NOTE: generate imbalanced data
    generator = ImbalancedDataGenerator(verbose=True, random_seed=0)
    (train_images, train_labels) = generator.transform(train_data, train_targets, imratio=imratio)
    (test_images, test_labels) = generator.transform(test_data, test_targets, imratio=0.5)

    # NOTE: data augmentations
    trainSet = ImageDataset(train_images, train_labels)
    testSet = ImageDataset(test_images, test_labels, mode="test")

    # NOTE: dataloaders
    sampler = DualSampler(trainSet, batch_size, sampling_rate=imratio)
    trainloader = torch.utils.data.DataLoader(trainSet, batch_size=batch_size, sampler=sampler, num_workers=2)
    testloader = torch.utils.data.DataLoader(testSet, batch_size=batch_size, shuffle=False, num_workers=2)

    # NOTE: network
    model = NeuralNetwork()
    weights_init(model)
    model = model.cuda()

    # NOTE: optimizer
    loss_fn = AUCMLoss(imratio=imratio)
    steps_per_epoch = len(trainloader)
    total_steps = max(1, total_epochs * steps_per_epoch)
    noise_scale = math.sqrt(total_steps * math.log(1 / delta)) / (n * epsilon)
    sigma_w = clip_w * noise_scale
    sigma_v = clip_v * noise_scale
    optimizer = DPSGDA(
        model.parameters(),
        loss_fn=loss_fn,
        lr_w=lr_w,
        lr_v=lr_v,
        sigma_w=sigma_w,
        sigma_v=sigma_v,
        clip_w=clip_w,
        clip_v=clip_v,
    )

    test_log = []
    acc_log = []
    for epoch in range(total_epochs):

        model.train()
        for i, (data, targets) in enumerate(trainloader):
            data, targets = data.cuda(), targets.cuda()

            optimizer.zero_grad()
            loss = loss_fn(torch.sigmoid(model(data)), targets)
            loss.backward()
            optimizer.step()

        # NOTE: evaluation on train & test sets
        model.eval()

        test_pred_list = []
        test_true_list = []
        correct = 0
        total = 0
        for test_data, test_targets in testloader:
            test_data = test_data.cuda()
            logits = model(test_data)
            probs = torch.sigmoid(logits).cpu()
            preds = (probs >= 0.5).int()
            correct += (preds.view(-1) == test_targets).sum().item()
            total += test_targets.size(0)
            test_pred_list.append(logits.cpu().detach().numpy())
            test_true_list.append(test_targets.numpy())
        test_true = np.concatenate(test_true_list)
        test_pred = np.concatenate(test_pred_list)
        val_auc = auc_roc_score(test_true, test_pred)
        test_acc = correct / total if total else 0.0
        model.train()

        # NOTE: print results
        print(
            "epoch: %s, test_auc: %.4f, test_acc: %.4f, lr_w: %.4f, lr_v: %.4f"
            % (epoch, val_auc, test_acc, optimizer.lr_w, optimizer.lr_v)
        )
        test_log.append(val_auc)
        acc_log.append(test_acc)

    print("best auc: %.4f" % max(np.array(test_log)))
    print("best acc: %.4f" % max(np.array(acc_log)))


parser = argparse.ArgumentParser(description='auc maximization.')
parser.add_argument('--epsilon', default=0.5, type=float, help='Param of differential privacy')
parser.add_argument("--clip_w", default=1.0, type=float, help="clip threshold for model gradients")
parser.add_argument("--clip_v", default=1.0, type=float, help="clip threshold for dual gradients")
parser.add_argument("--lr_w", default=0.2, type=float, help="learning rate for model parameters")
parser.add_argument("--lr_v", default=0.2, type=float, help="learning rate for dual variables")
parser.add_argument("--imratio", default=0.1, type=float, help="imratio")
parser.add_argument("--batch_size", default=2048, type=float, help="batch size")
parser.add_argument("--seed", default=0, type=int, help="seed")
parser.add_argument("--delta", default=1e-4, type=float, help="delta")
parser.add_argument("--total_epochs", default=80, type=int, help="epoch")

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    train(args)
