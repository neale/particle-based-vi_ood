import os
import wandb
import torch
import datagen
import numpy as np
from scipy.stats import entropy as entropy_fn
from sklearn.metrics import roc_auc_score
from utils import Averager, rbf_fn, insert_items, extract_parameters, kernel_cka
from ood_fns import auc_score, uncertainty, MattLoss


class Ensemble(object):

    def __init__(self,
        dataset,
        base_model_fn,
        num_particles=10,
        resume=False,
        resume_epoch=None,
        resume_lr=1e-4):
        
        self.dataset = dataset
        self.num_particles = num_particles
        print ("running {}-ensemble on {}".format(num_particles, dataset))

        self._use_wandb = False
        self._save_model = False

        if self.dataset == 'mnist':
            self.train_loader, self.test_loader, self.val_loader = datagen.load_mnist(split=True)
        elif self.dataset == 'cifar10':
            self.train_loader, self.test_loader, self.val_loader, = datagen.load_cifar10(split=True)
        else:
            raise NotImplementedError
        
        models = [base_model_fn(num_classes=6).cuda() for _ in range(num_particles)]
        
        self.models = models
        self.optimizer = [torch.optim.Adam(
            m.parameters(), lr=1e-3, weight_decay=1e-4) for m in models]
        self.schedulers = [torch.optim.lr_scheduler.StepLR(o, 100, .1) for o in self.optimizer]
        
        if resume:
            print ('resuming from epoch {}'.format(resume_epoch))
            d = torch.load('saved_models/{}/{}2/model_epoch_{}.pt'.format(
                self.dataset,
                model_id,
                resume_epoch))
            for model, sd in zip(self.models, d['models']):
                model.load_state_dict(sd)
            self.param_set = d['params']
            self.optimizer = [torch.optim.Adam(
                model.parameters(), lr=resume_lr, weight_decay=1e-4) for model in models]
            self.start_epoch = resume_epoch
        else:
            self.start_epoch = 0
        
        loss_type = 'ce'
        if loss_type == 'ce':
            self.loss_fn = torch.nn.CrossEntropyLoss()
        elif loss_type == 'kliep':
            self.loss_fn = MattLoss().get_loss_dict()['kliep']
        
        if self._use_wandb:
            wandb.init(project="open-category-experiments",
                       name="MLE CIFAR")
            for model in models:
                wandb.watch(model)
            config = wandb.config
            config.algo = 'ensemble'
            config.dataset = dataset
            config.kernel_fn = 'none'
            config.num_particles = num_particles
            config.loss_fn = loss_type

    def test(self, test_loader, eval_loss=True):
        for model in self.models:
            model.eval()
        correct = 0
        test_loss = 0
        outputs_all = []
        for i, (inputs, targets) in enumerate(test_loader):
            preds = []
            loss = 0
            inputs = inputs.cuda()
            targets = targets.cuda()
            for model in self.models:
                outputs = model(inputs)
                if eval_loss:
                    loss += self.loss_fn(outputs, targets)
                else:
                    loss += 0
                preds.append(torch.nn.functional.softmax(outputs, dim=-1))

            pred = torch.stack(preds)
            outputs_all.append(pred)
            preds = pred.mean(0)
            vote = preds.argmax(-1).cpu()
            correct += vote.eq(targets.cpu().data.view_as(vote)).float().cpu().sum()
            test_loss += (loss / self.num_particles)
        outputs_all = torch.cat(outputs_all, dim=1)
        test_loss /= i
        correct /= len(test_loader.dataset)
        for model in self.models:
            model.train()
        return outputs_all, (test_loss, correct)

    def train(self, epochs):
        for epoch in range(self.start_epoch, epochs):
            loss_epoch = 0
            for (inputs, targets) in self.train_loader:
                outputs = []
                neglogp = []
                for i, (model, optim) in enumerate(zip(self.models, self.optimizer)):
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                    output = model(inputs)
                    outputs.append(output)
                    loss = self.loss_fn(outputs[-1], targets)
                    neglogp.append(loss)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                
                loss_step = torch.tensor(neglogp).mean()
                loss_epoch += loss_step 

            loss_epoch /= self.num_particles
            print ('Train Epoch {} [cum loss: {}]\n'.format(epoch, loss_epoch))
            
            if epoch % 1 == 0:
                with torch.no_grad():
                    outputs, stats = self.test(self.val_loader)
                    outputs2, _ = self.test(self.test_loader, eval_loss=False)
                test_loss, correct = stats
                print ('Test Loss: {}'.format(test_loss))
                print ('Test Acc: {}%'.format(correct*100))
                uncertainties = uncertainty(outputs)
                entropy, variance = uncertainties
                uncertainties2 = uncertainty(outputs2)
                entropy2, variance2 = uncertainties2
            
                auc_entropy = auc_score(entropy, entropy2)
                print ('Test AUC Entropy: {}'.format(auc_entropy))
                auc_variance = auc_score(variance, variance2)
                print ('Test AUC Variance: {}'.format(auc_variance))
                
                if self._use_wandb:
                    wandb.log({"Test Loss": test_loss})
                    wandb.log({"Train Loss": loss_epoch})
                    wandb.log({"Test Acc": correct*100})
                    wandb.log({"Test AUC (entropy)": auc_entropy})
                    wandb.log({"Test AUC (variance)": auc_variance})
                
                if self._save_model:
                    params = {'models': [m.state_dict() for m in self.models],
                              'optimizer': [o.state_dict() for o in self.optimizer]}
                    save_dir = 'saved_models/{}/{}2/'.format(self.dataset, model_id)
                    fn = 'model_epoch_{}.pt'.format(epoch)
                    print ('saving model: {}'.format(fn))
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save(params, save_dir+fn)
            print ('*'*86)
            for scheduler in self.schedulers:
                scheduler.step()


if __name__ == '__main__':
    # model_id = 'resnet'
    # model_id = 'densenet'
    model_id = 'lenet'

    if model_id == 'resnet':
        #from preact_resnet import PreActResNet18 as base_model
        #from preact_resnet import PreActResNet50 as base_model
        from preact_resnet import PreActResNet101 as base_model
    elif model_id == 'densenet':
        from densenet import DenseNet121 as base_model
    elif model_id == 'lenet':    
        from lenet import LeNet as base_model
   
    resume = False
    resume_epoch = 0
    resume_lr = 1e-4
    runner = Ensemble(
        dataset='mnist',
        num_particles=2,
        base_model_fn=base_model,
        resume=resume,
        resume_epoch=resume_epoch,
        resume_lr=resume_lr)

    runner.train(300)
