import os
import torch
import datagen
import numpy as np
from scipy.stats import entropy as entropy_fn
from sklearn.metrics import roc_auc_score
from utils import Averager, rbf_fn, insert_items, extract_parameters, kernel_cka
from ood_fns import auc_score, uncertainty


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

        if self.dataset == 'mnist':
            self.train_loader, self.test_loader, self.val_loader = datagen.load_mnist(split=True)
        elif self.dataset == 'cifar10':
            self.train_loader, self.test_loader, self.val_loader, = datagen.load_cifar10(split=True)
        else:
            raise NotImplementedError
        
        if kernel_fn == 'rbf':
            self.kernel = rbf_fn
            return_activations = False
        elif kernel_fn == 'cka':
            self.kernel = kernel_cka
            return_activations = True
        else:
            raise NotImplementedError

        models = [base_model_fn(
                      num_classes=6,
                      return_activations=return_activations).cuda()
                  for _ in range(num_particles)]
        
        self.models = models
        param_set, state_dict = extract_parameters(self.models)
        
        self.state_dict = state_dict
        self.param_set = torch.nn.Parameter(param_set.clone(), requires_grad=True)

        self.optimizer = torch.optim.Adam([{'params': self.param_set,
                                            'lr':1e-3,
                                            'weight_decay':1e-4}])
        
        if resume:
            print ('resuming from epoch {}'.format(resume_epoch))
            d = torch.load('saved_models/{}/{}2/model_epoch_{}.pt'.format(
                self.dataset,
                model_id,
                resume_epoch))
            for model, sd in zip(self.models, d['models']):
                model.load_state_dict(sd)
            self.param_set = d['params']
            self.state_dict = d['state_dict']
            self.optimizer = torch.optim.Adam([{'params': self.param_set,
                                                'lr': resume_lr,
                                                'weight_decay': 1e-4}])
            self.start_epoch = resume_epoch
        else:
            self.start_epoch = 0
        
        self.activation_length = self.models[0].activation_length
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

    def svgd_grad(self, loss_grad, params):
        """
        Compute particle gradients via SVGD, empirical expectation
        evaluated by splitting half of the sampled batch. 
        """
        num_particles = params.shape[0]
        params2 = params.detach().requires_grad_(True)
        # kernel_weight, kernel_grad = self.kernel(params2, params, self.kernel_width)
        for i in range(num_particles):
            for j in range(num_particles):
                if i == j:
                    continue
                print (params[i].shape)
                k, _ = self.kernel(params[i], params[j])
                print (k.shape)
        if kernel_grad is None:
            kernel_grad = torch.autograd.grad(kernel_weight.sum(), params2)[0]

        kernel_logp = torch.matmul(kernel_weight.t().detach(),
                                   loss_grad) / num_particles 
        grad = kernel_logp - kernel_grad.mean(0)
        return grad

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
                #if self.kernel_fn == 'cka':
                   # outputs, _ = outputs
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
                activations = torch.zeros(
                    len(self.models),
                    len(targets),
                    self.activation_length).cuda()
                neglogp = torch.zeros(self.num_particles)
                insert_items(self.models, self.param_set, self.state_dict)
                neglogp_grads = torch.zeros_like(self.param_set)
                for i, model in enumerate(self.models):
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                    output, activation = model(inputs)
                    outputs.append(output)
                    activations[i, :, :] = activation
                    loss = self.loss_fn(outputs[-1], targets)
                    grad = torch.autograd.grad(loss.sum(), activation)[0]
                    print (grad)
                    print (grad.shape)
                    print (torch.count_nonzero(grad), np.prod(grad.shape))
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
            
            if epoch % 1 == 0:
                insert_items(self.models, self.param_set, self.state_dict)
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
                auc_variance = auc_score(variance, variance2)

                print ('Test AUC Entropy: {}'.format(auc_entropy))
                print ('Test AUC Variance: {}'.format(auc_variance))

            
                params = {'params': self.param_set,
                          'state_dict': self.state_dict,
                          'models': [m.state_dict() for m in self.models],
                          'optimizer': self.optimizer.state_dict()}
                save_dir = 'saved_models/{}/{}2/'.format(self.dataset, model_id)
                fn = 'model_epoch_{}.pt'.format(epoch)
                print ('saving model: {}'.format(fn))
                os.makedirs(save_dir, exist_ok=True)
                torch.save(params, save_dir+fn)
            print ('*'*86)


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
    runner = ParticleVI(
        algo='svgd',
        dataset='mnist',
        kernel_fn = 'cka',
        num_particles=10,
        base_model_fn=base_model,
        resume=resume,
        resume_epoch=resume_epoch,
        resume_lr=resume_lr)

    runner.train(1000)
