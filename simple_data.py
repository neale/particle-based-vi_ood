import torch
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns


def generate_regression_data(n_train, n_test):
    x_train1 = torch.linspace(-6, -2, n_train//2).view(-1, 1)
    x_train2 = torch.linspace(2, 6, n_train//2).view(-1, 1)
    x_train3 = torch.linspace(-2, 2, 4).view(-1, 1)
    x_train = torch.cat((x_train1, x_train2, x_train3), dim=0)
    y_train = -(1 + x_train) * torch.sin(1.2*x_train) 
    y_train = y_train + torch.ones_like(y_train).normal_(0, 0.04)

    x_test = torch.linspace(-6, 6, n_test).view(-1, 1)
    y_test = -(1 + x_test) * torch.sin(1.2*x_test) 
    y_test = y_test + torch.ones_like(y_test).normal_(0, 0.04)
    return (x_train, y_train), (x_test, y_test)

def plot_regression(models, data, epoch, tag):
    sns.set_style('darkgrid')
    gt_x = torch.linspace(-6, 6, 500).view(-1, 1).cpu()
    gt_y = -(1+gt_x) * torch.sin(1.2*gt_x) 
    (x_train, y_train), (x_test, y_test) = data
    outputs = []
    for model in models:
        outputs.append(model(x_test.cuda()).detach().cpu())
    outputs = torch.stack(outputs)
    mean = outputs.mean(0).squeeze()
    std = outputs.std(0).squeeze()
    x_test = x_test.cpu().numpy()
    x_train = x_train.cpu().numpy()

    plt.fill_between(x_test.squeeze(), mean.T+2*std.T, mean.T-2*std.T, alpha=0.5)
    plt.plot(gt_x, gt_y, color='red', label='ground truth')
    plt.plot(x_test, mean.T, label='posterior mean', alpha=0.9)
    plt.scatter(x_train, y_train.cpu().numpy(),color='r', marker='+',
        label='train pts', alpha=1.0, s=50)
    plt.legend(fontsize=14, loc='best')
    plt.ylim([-6, 8])
    os.makedirs('plots/', exist_ok=True)
    plt.savefig('plots/{}-conf-regression_{}.png'.format(tag, epoch))
    plt.close('all') 


def generate_classification_data(
    n_samples=200,
    #means=[(2., 2.), (-2., 2.), (2., -2.), (-2., -2.)]):
    means=[(2., 2.), (-2., -2.)]):

    data = torch.zeros(n_samples, 2)
    labels = torch.zeros(n_samples)
    size = n_samples//len(means)
    for i, (x, y) in enumerate(means):
        dist = torch.distributions.Normal(torch.tensor([x, y]), .3)
        samples = dist.sample([size])
        data[size*i:size*(i+1)] = samples
        labels[size*i:size*(i+1)] = torch.ones(len(samples)) * i
    
    return data, labels.long()

def plot_classification(models, epoch, tag):
    x = torch.linspace(-10, 10, 100)
    y = torch.linspace(-10, 10, 100)
    gridx, gridy = torch.meshgrid(x, y)
    grid = torch.stack((gridx.reshape(-1), gridy.reshape(-1)), -1)
   
    outputs = []
    for model in models: 
        outputs.append(model(grid.cuda()).detach().cpu())
    outputs = torch.stack(outputs)
    outputs = torch.nn.functional.softmax(outputs, -1).detach()  # [B, D]
    mean_outputs = outputs.mean(0).cpu()  # [B, D]
    std_outputs = outputs.std(0).cpu()
    conf_std = std_outputs.max(-1)[0] * 1.94
    labels = mean_outputs.argmax(-1)
    data, _ = generate_classification_data(n_samples=400) 
    
    p1 = plt.scatter(grid[:, 0].cpu(), grid[:, 1].cpu(), c=conf_std,
        cmap='rainbow')
    p2 = plt.scatter(data[:, 0].cpu(), data[:, 1].cpu(), c='black', alpha=0.1)
    cbar = plt.colorbar(p1)
    cbar.set_label("confidance (std)")
    os.makedirs('plots/', exist_ok=True)
    plt.savefig('plots/{}-conf-std_{}.png'.format(epoch, tag))
    plt.close('all')

