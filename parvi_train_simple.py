import os
import wandb
import torch
import datagen
import numpy as np
from utils import Averager, rbf_fn, insert_items, extract_parameters
import simple_data as toy


class ParticleVI(object):

    def __init__(self,
        algo,
        dataset,
        kernel_fn,
        base_model_fn,
        num_particles=10,
        resume=False,
        resume_epoch=None,
        resume_lr=1e-4):
        
        self.algo = algo
        self.dataset = dataset
        self.kernel_fn = kernel_fn
        self.num_particles = num_particles
        print ("running {} on {}".format(algo, dataset))
        
        if self.dataset == 'regression':
            self.data = toy.generate_regression_data(80, 200)
            (self.train_data, self.train_targets), (self.test_data, self.test_targets) = self.data
        elif self.dataset == 'classification':
            self.train_data, self.train_targets = toy.generate_classification_data(100)
            self.test_data, self.test_targets = toy.generate_classification_data(200)
        else:
            raise NotImplementedError
            
        
        if kernel_fn == 'rbf':
            self.kernel = rbf_fn
        else:
            raise NotImplementedError
        
        models = [base_model_fn().cuda() for _ in range(num_particles)]
        
        self.models = models
        param_set, state_dict = extract_parameters(self.models)
        
        self.state_dict = state_dict
        self.param_set = torch.nn.Parameter(param_set.clone(), requires_grad=True)

        self.optimizer = torch.optim.Adam([{'params': self.param_set,
                                            'lr':1e-3}])
        
        if self.dataset == 'regression':
            self.loss_fn = torch.nn.MSELoss()
        elif self.dataset == 'classification':
            self.loss_fn = torch.nn.CrossEntropyLoss()
        self.kernel_width_averager = Averager(shape=())
        
    def kernel_width(self, dist):
        """Update kernel_width averager and get latest kernel_width. """
        if dist.ndim > 1:
            dist = torch.sum(dist, dim=-1)
            assert dist.ndim == 1, "dist must have dimension 1 or 2."
        width, _ = torch.median(dist, dim=0)
        width = width / np.log(len(dist))
        self.kernel_width_averager.update(width)
        return self.kernel_width_averager.get()

    def rbf_fn(self, x, y):
        Nx = x.shape[0]
        Ny = y.shape[0]
        x = x.view(Nx, -1)
        y = y.view(Ny, -1)
        Dx = x.shape[1]
        Dy = y.shape[1]
        assert Dx == Dy
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # [Nx, Ny, D]
        dist_sq = torch.sum(diff**2, -1)  # [Nx, Ny]
        h = self.kernel_width(dist_sq.view(-1))
        kappa = torch.exp(-dist_sq / h)  # [Nx, Nx]
        kappa_grad = torch.einsum('ij,ijk->ijk', kappa,
                                  -2 * diff / h)  # [Nx, Ny, D]
        return kappa, kappa_grad

    def svgd_grad(self, loss_grad, params):
        """
        Compute particle gradients via SVGD, empirical expectation
        evaluated by splitting half of the sampled batch. 
        """
        num_particles = params.shape[0]
        params2 = params.detach().requires_grad_(True)
        kernel_weight, kernel_grad = self.rbf_fn(params2, params)
        if kernel_grad is None:
            kernel_grad = torch.autograd.grad(kernel_weight.sum(), params2)[0]

        kernel_logp = torch.matmul(kernel_weight.t().detach(),
                                   loss_grad) / num_particles 
        grad = kernel_logp - kernel_grad.mean(0)
        print (kernel_logp.norm(), kernel_grad.mean(0).norm())
        return grad

    def test(self, eval_loss=True):
        for model in self.models:
            model.eval()
        correct = 0
        test_loss = 0
        preds = []
        loss = 0
        test_data = self.test_data.cuda()
        test_targets = self.test_targets.cuda()
        for model in self.models:
            outputs = model(test_data)
            if eval_loss:
                loss += self.loss_fn(outputs, test_targets)
            else:
                loss += 0
            preds.append(outputs)

        preds = torch.stack(preds)
        p_mean = preds.mean(0)
        if self.dataset == 'classification':
            preds = torch.nn.functional.softmax(preds, dim=-1)
            preds = preds.mean(0)
            vote = preds.argmax(-1).cpu()
            correct = vote.eq(test_targets.cpu().data.view_as(vote)).float().cpu().sum()
            correct /= len(test_targets)
        else:
            correct = 0
            test_loss += (loss / self.num_particles)
        outputs_all = preds
        test_loss /= len(self.models)
        for model in self.models:
            model.train()
        return outputs_all, (test_loss, correct)

    def train(self, epochs):
        for epoch in range(0, epochs):
            loss_epoch = 0
            neglogp = torch.zeros(self.num_particles)
            insert_items(self.models, self.param_set, self.state_dict)
            neglogp_grads = torch.zeros_like(self.param_set)
            outputs = []
            for i, model in enumerate(self.models):
                train_data = self.train_data.cuda()
                train_targets = self.train_targets.cuda()
                output = model(train_data)
                outputs.append(output)
                loss = self.loss_fn(outputs[-1], train_targets)
                loss.backward()
                neglogp[i] = loss
                g = []
                for name, param in model.named_parameters():
                    g.append(param.grad.view(-1))
                neglogp_grads[i] = torch.cat(g)
                model.zero_grad()
            
            par_vi_grad = self.svgd_grad(neglogp_grads, self.param_set)
            self.optimizer.zero_grad()
            self.param_set.grad = par_vi_grad
            self.optimizer.step()
    
            loss_step = neglogp.mean()
            loss_epoch += loss_step 

            loss_epoch /= self.num_particles
            print ('Train Epoch {} [cum loss: {}]\n'.format(epoch, loss_epoch))
            
            if epoch % 100 == 0:
                insert_items(self.models, self.param_set, self.state_dict)
                with torch.no_grad():
                    outputs, stats = self.test(eval_loss=False)
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
    runner = ParticleVI(
        algo='svgd',
        dataset=dataset,
        kernel_fn = 'rbf',
        num_particles=100,
        base_model_fn=base_model,
        resume=resume,
        resume_epoch=resume_epoch,
        resume_lr=resume_lr)

    runner.train(10000)
