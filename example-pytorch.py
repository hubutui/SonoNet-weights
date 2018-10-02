#!/bin/env python3

import numpy as np
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn import metrics
import configargparse
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import SonoNet


def test(model, device, test_loader):
    model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        y_pred = []
        y_true = []
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = torch.mean(output.view(output.size(0), output.size(1), -1), dim=2)
            test_loss += F.cross_entropy(output, target)
            output = F.softmax(output, dim=1)
            confidence, pred = output.max(1)
            print('confidence: {}, prediction: {}, ground truth: {}'.format(confidence.cpu().numpy(), pred.cpu().numpy(), target.cpu().numpy()))
            y_pred += pred.data.tolist()
            y_true += target.data.tolist()
            correct += pred.eq(target.view_as(pred)).sum().item()

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print(metrics.classification_report(np.asarray(y_true), np.asarray(y_pred)))
    print('confusion matrix: \n', metrics.confusion_matrix(np.asarray(y_true), np.asarray(y_pred)))
    print('\n')


def main():
    parser = configargparse.ArgParser()
    parser.add('--device', default='auto', choices=['auto', 'gpu', 'cpu'],
               help='device to train (default: auto)')
    parser.add('--model', default='SonoNet32', choices=['SonoNet16', 'SonoNet32', 'SonoNet64', 'SmallNet'],
               help='model to evaluate, default: SonoNet32')
    parser.add('--test-dir', default='example_images/test', metavar='DIR',
               help='root directory of test data, default: example_images/test')

    args = parser.parse_args()
    use_cuda = not args.device != 'auto' and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    mean = (0.456,)
    std = (0.225,)

    tf = transforms.Compose([
        transforms.Resize((224, 288)),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.Lambda(lambda x: x*255.)
    ])

    testset = ImageFolder(args.test_dir, transform=tf)
    testloader = DataLoader(testset, shuffle=False)
    if args.model == 'SonoNet16':
        model = SonoNet.SonoNet16()
    elif args.model == 'SonoNet32':
        model = SonoNet.SonoNet32()
    elif args.model == 'SonoNet64':
        model = SonoNet.SonoNet64()
    elif args.model == 'SmallNet':
        model = SonoNet.SmallNet()
    else:
        raise ValueError('No such network or model')

    print("Predictions with GPU: ", use_cuda)
    print("Predictions using {}".format(args.model))
    model.load_state_dict(torch.load(args.model + '.pytorch.pth'))
    start_time = time.time()
    test(model, device, testloader)
    total_time = time.time() - start_time
    print('FPS: ', total_time/14)

if __name__=='__main__':
    main()
