import os
import sys
from path_config import *
sys.path.append(src_root_path)
import time
import tqdm
import torch
import argparse
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

import warnings
warnings.filterwarnings("ignore")


from src_ours.utils import create_log_dir, set_seed
from src_ours.eval import eval_single_dataset_preprocess_mapping_head, eval_single_dataset_preprocess_head
from src_ours.merging_model import ModelWrapper, AlphaWrapper, make_functional
from src_ours.ties_merging_utils import *
from src_ours.task_vectors import TaskVector
from datasets_.registry import get_dataset
from datasets_.common import maybe_dictionarize, get_dataloader_shuffle


def get_merged_model():
    # Create the task vectors
    if args.method_name in ['ties_merging', 'tw_adamergingpp', 'lw_adamergingpp']:

        ft_checks = [torch.load(checkpoint_path + args.model_name + '/' + dataset_name + '/finetuned.pt').state_dict() for dataset_name in learn_datasets]
        ptm_check = torch.load(args.base_checkpoint).state_dict()
        check_parameterNamesMatch(ft_checks + [ptm_check])
        remove_keys = []

        flat_ft = torch.vstack([state_dict_to_vector(check, remove_keys) for check in ft_checks])
        flat_ptm = state_dict_to_vector(ptm_check, remove_keys)

        tv_flat_checks = flat_ft - flat_ptm
        assert check_state_dicts_equal(vector_to_state_dict(flat_ptm, ptm_check, remove_keys), ptm_check)
        assert all([check_state_dicts_equal(vector_to_state_dict(flat_ft[i], ptm_check, remove_keys), ft_checks[i]) for i in range(len(ft_checks))])
        selected_entries, merged_tv = ties_merging_split(tv_flat_checks, reset_thresh=20, merge_func="dis-sum", )
        
        ties_task_vectors = []
        for vector_ in selected_entries:
            t_state_dict = vector_to_state_dict(vector_, ptm_check, remove_keys=remove_keys)
            ref_model = torch.load(args.base_checkpoint)
            ref_model.load_state_dict(t_state_dict, strict=False)
            ties_task_vectors.append(ref_model.state_dict())

    elif args.method_name in ['weight_averaging', 'task_arithmetic', 'tw_adamerging', 'lw_adamerging']:
        # Task Vector
        task_vectors = [TaskVector(args.base_checkpoint, checkpoint_path + args.model_name + '/' + dataset_name + '/finetuned.pt') for dataset_name in learn_datasets]

    else:
        print('method name error!')
        exit(-1)

    base_model = torch.load(args.base_checkpoint)
    bsae_model_dic = base_model.state_dict()

    model = ModelWrapper(base_model)
    model = model.to(args.device)
    _, layer_names = make_functional(model)

    paramslist = []
    paramslist += [tuple(v.detach().requires_grad_().cpu() for _, v in bsae_model_dic.items())] # pretrain
    if args.method_name in ['ties_merging', 'tw_adamergingpp', 'lw_adamergingpp']:
        paramslist += [tuple(v.detach().requires_grad_().cpu() for _, v in sd.items())  for i, sd in enumerate(ties_task_vectors)] # task vectors
    elif args.method_name in ['weight_averaging', 'task_arithmetic', 'tw_adamerging', 'lw_adamerging']:
        paramslist += [tuple(v.detach().requires_grad_().cpu() for _, v in sd.vector.items()) for i, sd in enumerate(task_vectors)]  # task vectors

    torch.cuda.empty_cache()
    alpha_model = AlphaWrapper(paramslist, model, layer_names, learn_datasets, args)

    return base_model, alpha_model


def train():
    # preload dataloaders, batches, encoders for different datasets
    all_finetuned_encoders = []
    all_datasets = []
    all_dataloaders = []

    for dataset_name in learn_datasets:
        try:
            finetuned = torch.load(checkpoint_path + args.model_name + '/' + dataset_name + '/finetuned.pt')
            finetuned = finetuned.to(args.device)
            finetuned.eval()
            all_finetuned_encoders.append(finetuned)

            dataset = get_dataset(dataset_name, base_model.val_preprocess, location=args.data_location, batch_size=16)
            all_datasets.append(dataset)

            dataloader = get_dataloader_shuffle(dataset)
            all_dataloaders.append(iter(dataloader))
        except:
            print("Error ckpt in {}".format(checkpoint_path + args.model_name + '/' + dataset_name + '/finetuned.pt'))

    all_batches = []
    all_learn_datasets = []

    for i in tqdm.tqdm(range(batch_per_iter), desc='preload 100 batchs:', ncols=100):
        all_batches += [next(all_dataloaders[index]) for index in range(len(learn_datasets))]
        all_learn_datasets += [learn_datasets[i] for i in range(len(learn_datasets))]

    # start to train the module
    distance_list = []
    for i in tqdm.tqdm(range(batch_per_iter * len(learn_datasets)), desc='training on these 100 batchs:', ncols=100):
        batch = all_batches[i]
        dataset_name = all_learn_datasets[i]

        data = maybe_dictionarize(batch)
        x = data['images'].to(args.device)
        
        _, revised_feature, _, kl_loss = alpha_model(x, dataset_name)

        finetuned_encoder = all_finetuned_encoders[i % len(learn_datasets)]
        finetuned_features = finetuned_encoder(x).detach()

        loss_dic = loss_func(revised_feature, finetuned_features)
        loss = loss_dic + args.KL_weight * kl_loss

        distance_list.append(loss_dic)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def eval():
    Total_ACC = 0.
    ACC_per_Dataset = []
    Dataset_names = []

    for dataset_name in exam_datasets:
        image_encoder = alpha_model.get_image_encoder()
        classification_head = alpha_model.get_classification_head(dataset_name)
        up_proj, linear_mean, linear_var = alpha_model.get_feature_mapping_to_head(dataset_name)
        metrics = eval_single_dataset_preprocess_mapping_head(image_encoder, classification_head, dataset_name, args, up_proj, linear_mean, linear_var)
        test_acc = metrics['top1']
        
        Total_ACC += test_acc
        ACC_per_Dataset.append(metrics['top1'])
        Dataset_names.append(dataset_name)

    Val_ACC = Total_ACC / len(learn_datasets)
    return Val_ACC, ACC_per_Dataset, Dataset_names


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-location", type=str, default=os.path.expanduser('~/data'), help="The root directory for the datasets.",)
    parser.add_argument("--eval-datasets", default=None, type=lambda x: x.split(","), help="Which datasets to use for evaluation. Split by comma, e.g. MNIST,EuroSAT. ")
    parser.add_argument("--train-dataset", default=None, type=lambda x: x.split(","), help="Which dataset(s) to patch on.")
    parser.add_argument("--openclip-cachedir", type=str,default='./.cache/open_clip',help='Directory for caching models from OpenCLIP')
    parser.add_argument("--cache_dir", type=str,default='./.cache/open_clip',help='Directory for caching models from OpenCLIP')
    
    parser.add_argument("--method-name", type=str, default="weight_averaging", 
                        choices=['weight_averaging', 'task_arithmetic', 'tw_adamerging', 'lw_adamerging', 'ties_merging', 'tw_adamergingpp', 'lw_adamergingpp'])
    
    parser.add_argument("--model-name", type=str, default="ViT-B-32")
    parser.add_argument("--loss_func", type=str, default="L1_loss", choices=["L1_loss", "mse_Loss", "smoothL1_Loss", "CosineSimilarity"])
    parser.add_argument("--batch-size",type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--hidden_layer", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--KL_weight", type=float, default=1e-4)
    parser.add_argument("--iter_number", type=int, default=5)

    parser.add_argument("--one", type=bool, default=False)
    parser.add_argument("--save_ckpt", type=bool, default=False)
    parsed_args = parser.parse_args()

    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return parsed_args
    
    
if __name__ == '__main__':
    args = get_args()
    set_seed(args.seed)

    exam_datasets =  ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD']
    learn_datasets = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD']
    
    iterations_number = args.iter_number
    eval_iterations = 1
    batch_per_iter = 100

    args.model = args.model_name
    args.data_location = dataset_path
    args.output_path = output_path
    args.save = checkpoint_path + args.model_name

    args.logs_path = args.output_path + "ProbSurgery/" + args.model_name
    args.base_checkpoint = checkpoint_path + args.model_name + '/zeroshot.pt'
    log = create_log_dir(args.logs_path, '{}_{}_{}_{}_{}.txt'.format(args.method_name, 
                                                               time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time())),
                                                               args.iter_number,
                                                               args.loss_func,
                                                               args.hidden_layer))
    
    log.info(str(args))

    # =============== start merge ===================
    base_model, alpha_model = get_merged_model()

    optimizer = torch.optim.Adam(alpha_model.collect_trainable_params(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.)
    if args.loss_func == 'L1_loss':
        loss_func = torch.nn.L1Loss()
    elif args.loss_func == 'mse_Loss':
        loss_func = torch.nn.MSELoss()
    elif args.loss_func == 'smoothL1_Loss':
        loss_func = torch.nn.SmoothL1Loss()
    elif args.loss_func == 'CosineSimilarity':
        loss_func = torch.nn.CosineSimilarity(dim=1)
    else:
        log.info("Error input loss function...")

    log.info(f"================= (merge w/ prob_surgery) Base method is {args.method_name} ==============")
    best_acc = 0.0
    best_acc_dataset = []

    for i in range(args.iter_number):
        train()

        if i % eval_iterations == 0:
            Val_ACC, ACC_per_Dataset, Dataset_names = eval()

            log.info(f"Results on step {str(batch_per_iter * (i+1))}")
            log.info(''.join(f"{s:<10}" for s in Dataset_names))  # dataset name    
            log.info(''.join(f"{num:<10.5f}" for num in ACC_per_Dataset))  # the corresponding val_acc
            log.info(f'=== Avg acc: {str(Val_ACC)[:6]} \n')

            if best_acc < Val_ACC:
                best_acc = Val_ACC
                best_acc_dataset = ACC_per_Dataset
                if args.save_ckpt:
                    ckpt_path = args.logs_path + '/alpha_model_' \
                                            + args.method_name + "_" + str(args.iter_number) \
                                            + "_" + args.loss_func \
                                            + "_" + str(args.hidden_layer) + '.pth'
                    
                    torch.save(alpha_model, ckpt_path)
                    log.info("=== Save the best model to {}".format(ckpt_path))
                
                
    log.info('=== The best val accruacy is {}'.format(str(best_acc)[:7]))
    log.info("=== " + ''.join(f"{num:<10.5f}" for num in best_acc_dataset) + "\n")


    for key, value in vars(args).items():
        log.info(f"{key}: {value}")