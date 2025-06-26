import torch
import math
import torch.nn.functional as F

from src_ours.heads import get_classification_head as get_finetuned_classification_head
from src_ours.merging_cofficient import get_merging_cofficients


# Utilities to make nn.Module functional
def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])

def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)

def make_functional(mod):
    orig_params = tuple(mod.parameters())
    # Remove all the parameters in the model
    names = []
    for name, p in list(mod.named_parameters()):
        del_attr(mod, name.split("."))
        names.append(name)
    return orig_params, names

def load_weights(mod, names, params):
    for name, p in zip(names, params):
        set_attr(mod, name.split("."), p)

def softmax_entropy(x):
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


class ModelWrapper(torch.nn.Module):
    def __init__(self, model, initial_weights=None):
        super(ModelWrapper, self).__init__()
        self.model = model

        # Note: modified. Get rid of the language part.
        if hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images):
        features = self.model(images)
        return features


class AlphaWrapper(torch.nn.Module):
    def __init__(self, paramslist, model, names, exam_datasets, args):
        super(AlphaWrapper, self).__init__()
        self.paramslist = paramslist
        self.model = model
        self.names = names
        self.exam_datasets = exam_datasets
        self.args = args

        ralpha = get_merging_cofficients(args.method_name, args.model_name)

        self.alpha = torch.Tensor(ralpha)

        self.non_linear_func = torch.nn.ReLU()

        if args.one:
            if args.model_name != "ViT-L-14":
                up_proj = torch.nn.Linear(512, args.hidden_layer, bias=False)
                linear_mean = torch.nn.Linear(args.hidden_layer, 512, bias=False)
                linear_var = torch.nn.Linear(args.hidden_layer, 512, bias=False)
            else:
                up_proj = torch.nn.Linear(768, args.hidden_layer, bias=False)
                linear_mean = torch.nn.Linear(args.hidden_layer, 768, bias=False)
                linear_var = torch.nn.Linear(args.hidden_layer, 768, bias=False)

            torch.nn.init.kaiming_uniform_(up_proj.weight, a=math.sqrt(5))
            torch.nn.init.zeros_(linear_mean.weight)
            torch.nn.init.zeros_(linear_var.weight)


            self.add_module('feature_mapping_to_head_up_proj', up_proj.to(args.device))
            self.add_module('feature_mapping_to_head_mean_proj', linear_mean.to(args.device))
            self.add_module('feature_mapping_to_head_var_proj', linear_var.to(args.device))

        else:
            for dataset_name in exam_datasets:
                # mapping
                # ViT-B/32 and ViT-B/16
                if args.model_name != "ViT-L-14":
                    up_proj = torch.nn.Linear(512, args.hidden_layer, bias=False)
                    linear_mean = torch.nn.Linear(args.hidden_layer, 512, bias=False)
                    linear_var = torch.nn.Linear(args.hidden_layer, 512, bias=False)
                else:
                    up_proj = torch.nn.Linear(768, args.hidden_layer, bias=False)
                    linear_mean = torch.nn.Linear(args.hidden_layer, 768, bias=False)
                    linear_var = torch.nn.Linear(args.hidden_layer, 768, bias=False)

                torch.nn.init.kaiming_uniform_(up_proj.weight, a=math.sqrt(5))
                torch.nn.init.zeros_(linear_mean.weight)
                torch.nn.init.zeros_(linear_var.weight)
                # torch.nn.init.kaiming_uniform_(linear_mean.weight)
                # torch.nn.init.kaiming_uniform_(linear_var.weight)

                self.add_module('feature_mapping_to_head_up_proj_{}'.format(dataset_name), up_proj.to(args.device))
                self.add_module('feature_mapping_to_head_mean_proj_{}'.format(dataset_name), linear_mean.to(args.device))
                self.add_module('feature_mapping_to_head_var_proj_{}'.format(dataset_name), linear_var.to(args.device))

        for dataset_name in exam_datasets:
            classification_head = get_finetuned_classification_head(args, dataset_name)
            layer_name = 'classifier_{}'.format(dataset_name)
            self.add_module(layer_name, classification_head.to(args.device))


        if self.alpha.size()[0] == 1: # task-wise merging
            params = tuple(sum(tuple(pi * alphai for pi, alphai in zip(p, self.alpha[0].cpu()))) for j, p in enumerate(zip(*self.paramslist)))
        else: # layer-wise merging
            params = tuple(sum(tuple(pi * alphai for pi, alphai in zip(p, self.alpha[j].cpu()))) for j, p in enumerate(zip(*self.paramslist)))

        params = tuple(p.cuda(0) for p in params)
        load_weights(self.model, self.names, params)


    def collect_trainable_params(self):
        trainable_params = []

        if self.args.one:
            up_proj = getattr(self, 'feature_mapping_to_head_up_proj')
            linear_mean = getattr(self, 'feature_mapping_to_head_mean_proj')
            linear_var = getattr(self, 'feature_mapping_to_head_var_proj')

            trainable_params.append(up_proj.weight)
            trainable_params.append(linear_mean.weight)
            trainable_params.append(linear_var.weight)

        else:
        # surgery parameter
            for dataset_name in self.exam_datasets:
                up_proj = getattr(self, 'feature_mapping_to_head_up_proj_{}'.format(dataset_name))
                linear_mean = getattr(self, 'feature_mapping_to_head_mean_proj_{}'.format(dataset_name))
                linear_var = getattr(self, 'feature_mapping_to_head_var_proj_{}'.format(dataset_name))

                trainable_params.append(up_proj.weight)
                trainable_params.append(linear_mean.weight)
                trainable_params.append(linear_var.weight)
        return trainable_params

    def get_classification_head(self, dataset_name):
        layer_name = 'classifier_{}'.format(dataset_name)
        classification_head = getattr(self, layer_name)
        return classification_head

    def get_feature_mapping_to_head(self, dataset_name):
        if self.args.one:
            up_proj = getattr(self, 'feature_mapping_to_head_up_proj')
            linear_mean = getattr(self, 'feature_mapping_to_head_mean_proj')
            linear_var = getattr(self, 'feature_mapping_to_head_var_proj')
            return up_proj, linear_mean, linear_var

        else:
            up_proj = getattr(self, 'feature_mapping_to_head_up_proj_{}'.format(dataset_name))
            linear_mean = getattr(self, 'feature_mapping_to_head_mean_proj_{}'.format(dataset_name))
            linear_var = getattr(self, 'feature_mapping_to_head_var_proj_{}'.format(dataset_name))
            return up_proj, linear_mean, linear_var

    def get_image_encoder(self):
        if self.alpha.size()[0] == 1: # task-wise merging
            params = tuple(sum(tuple(pi * alphai for pi, alphai in zip(p, self.alpha[0].cpu()))) for j, p in enumerate(zip(*self.paramslist)))
        else: # layer-wise merging
            params = tuple(sum(tuple(pi * alphai for pi, alphai in zip(p, self.alpha[j].cpu()))) for j, p in enumerate(zip(*self.paramslist)))

        params = tuple(p.cuda(0) for p in params)
        load_weights(self.model, self.names, params)
        return self.model
    
    def get_image_feature(self, inp):
        feature = self.model(inp)

        return feature
    
    def get_surgery_feature(self, inp, dataset_name):
        feature = self.model(inp).detach()

        feature0 = feature
        # feature bias
        if self.args.one:
            up_proj = getattr(self, 'feature_mapping_to_head_up_proj')
            linear_mean = getattr(self, 'feature_mapping_to_head_mean_proj')
        else:
            up_proj = getattr(self, 'feature_mapping_to_head_up_proj_{}'.format(dataset_name))
            linear_mean = getattr(self, 'feature_mapping_to_head_mean_proj_{}'.format(dataset_name))
        # linear_var = getattr(self, 'feature_mapping_to_head_var_proj_{}'.format(dataset_name))

        feature_sub = up_proj(feature)
        feature_sub = self.non_linear_func(feature_sub)
        mu = linear_mean(feature_sub)
        revised_feature = feature0 - mu

        return revised_feature

    
    def KL_reg(self, log_var, mu):
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        return kld_loss
    

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        # eps = torch.randn_like(std.unsqueeze(0).repeat(self.args.sample_number, 1, 1))
        return eps * std + mu


    def forward(self, inp, dataset_name):
        feature = self.model(inp).detach()

        feature0 = feature

        # feature bias
        if self.args.one:
            up_proj = getattr(self, 'feature_mapping_to_head_up_proj')
            linear_mean = getattr(self, 'feature_mapping_to_head_mean_proj')
            linear_var = getattr(self, 'feature_mapping_to_head_var_proj')
        else:
            up_proj = getattr(self, 'feature_mapping_to_head_up_proj_{}'.format(dataset_name))
            linear_mean = getattr(self, 'feature_mapping_to_head_mean_proj_{}'.format(dataset_name))
            linear_var = getattr(self, 'feature_mapping_to_head_var_proj_{}'.format(dataset_name))

        feature_sub = up_proj(feature)
        feature_sub = self.non_linear_func(feature_sub)
        mu = linear_mean(feature_sub)
        log_var = linear_var(feature_sub)

        z = self.reparameterize(mu, log_var)
        kl_loss = self.KL_reg(log_var=log_var, mu=mu)

        revised_feature = feature0 - z


        layer_name = 'classifier_{}'.format(dataset_name)
        classification_head = getattr(self, layer_name) 
        out = classification_head(revised_feature)

        return out, revised_feature, z, kl_loss
