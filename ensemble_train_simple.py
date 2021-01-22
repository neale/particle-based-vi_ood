import os
import wandb
import torch
import datagen
import numpy as np
from scipy.stats import entropy as entropy_fn
from sklearn.metrics import roc_auc_score
from utils import Averager, rbf_fn, insert_items, extract_parameters, kernel_cka
from ood_fns import auc_score, uncertainty, MattLoss
from simple_models import Regressor, Classifier
import simple_data as toy


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
        
        if self.dataset == 'regression':
            self.data = toy.generate_regression_data(80, 200)
            (self.train_data, self.train_targets), (self.test_data, self.test_targets) = self.data
        elif self.dataset == 'classification':
            self.train_data, self.train_targets = toy.generate_classification_data(100)
            self.test_data, self.test_targets = toy.generate_classification_data(200)
        else:
            raise NotImplementedError
        
        models = [base_model_fn().cuda() for _ in range(num_particles)]
        
        self.models = models
        self.optimizer = [torch.optim.Adam(m.parameters(), lr=1e-3) for m in models]
        
        self.start_epoch = 0
        
        if self.dataset == 'regression':
            self.loss_fn = torch.nn.MSELoss()
        elif self.dataset == 'classification':
            self.loss_fn = torch.nn.CrossEntropyLoss()
                
    def test(self, eval_loss=True):
        for model in self.models:
            model.eval()
        correct = 0
        test_loss = 0
        outputs_all = []
        preds = []
        loss = 0
        inputs = self.test_data.cuda()
        targets = self.test_targets.cuda()
        for model in self.models:
            outputs = model(inputs)
            if eval_loss:
                loss += self.loss_fn(outputs, targets)
            else:
                loss += 0
            preds.append(outputs)

        preds = torch.stack(preds)
        p_mean = preds.mean(0)

        if self.dataset == 'classification':
            preds = torch.nn.functional.softmax(preds, dim=-1)
            preds = preds.mean(0)
            vote = preds.argmax(-1).cpu()
            correct = vote.eq(targets.cpu().data.view_as(vote)).float().cpu().sum()
            correct /= len(targets)
        else:
            correct = 0
        outputs_all = preds
        test_loss = (loss / self.num_particles) / len(self.models)
        
        for model in self.models:
            model.train()
        return outputs_all, (test_loss, correct)

    def train(self, epochs):
        for epoch in range(self.start_epoch, epochs):
            loss_epoch = 0
            outputs = []
            neglogp = []
            for i, (model, optim) in enumerate(zip(self.models, self.optimizer)):
                inputs = self.train_data.cuda()
                targets = self.train_targets.cuda()
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
            
            if epoch % 100 == 0:
                with torch.no_grad():
                    outputs2, stats = self.test(eval_loss=False)
                test_loss, correct = stats
                print ('Test Loss: {}'.format(test_loss))
                print ('Test Acc: {}%'.format(correct*100))
                if self.dataset == 'regression':
                    toy.plot_regression(self.models, self.data, epoch)
                if self.dataset == 'classification':
                    toy.plot_classification(self.models, epoch)    
            print ('*'*86)


if __name__ == '__main__':
    model_id = 'classifier'
    
    if model_id == 'regressor':
        from simple_models import Regressor as base_model
        dataset = 'regression'
    elif model_id == 'classifier':
        from simple_models import Classifier as base_model
        dataset = 'classification'
   
    resume = False
    resume_epoch = 0
    resume_lr = 1e-4
    runner = Ensemble(
        dataset=dataset,
        num_particles=100,
        base_model_fn=base_model,
        resume=resume,
        resume_epoch=resume_epoch,
        resume_lr=resume_lr)

    runner.train(10000)
