import os
import sys
from path_config import *
sys.path.append(src_root_path)
import time
import tqdm
import torch
import argparse
import random
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

from src.utils import create_log_dir, set_seed
from src.eval import eval_single_dataset_preprocess_mapping_head, eval_single_dataset, eval_single_dataset_preprocess_head
from src.merging_model import ModelWrapper, AlphaWrapper, make_functional
from src.ties_merging_utils import *
from src.task_vectors import TaskVector
from datasets_.registry import get_dataset
from datasets_.common import maybe_dictionarize, get_dataloader_shuffle


def get_merged_model():
    # Create the task vectors
    if args.method_name in ['ties_merging', 'tw_adamergingpp', 'lw_adamergingpp']:

        ft_checks = [torch.load(checkpoint_path + args.model_name + '/' + dataset_name + '/finetuned.pt').state_dict() for dataset_name in exam_datasets]
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
        task_vectors = [TaskVector(args.base_checkpoint, checkpoint_path + args.model_name + '/' + dataset_name + '/finetuned.pt') for dataset_name in exam_datasets]

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
    alpha_model = AlphaWrapper(paramslist, model, layer_names, exam_datasets, args)

    return base_model, alpha_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-location", type=str, default=os.path.expanduser('~/data'), help="The root directory for the datasets.",)
    parser.add_argument("--eval-datasets", default=None, type=lambda x: x.split(","), help="Which datasets to use for evaluation. Split by comma, e.g. MNIST,EuroSAT. ")
    parser.add_argument("--train-dataset", default=None, type=lambda x: x.split(","), help="Which dataset(s) to patch on.",)
    parser.add_argument("--batch-size",type=int, default=256)
    parser.add_argument("--openclip-cachedir", type=str,default='./.cache/open_clip',help='Directory for caching models from OpenCLIP')
    
    parser.add_argument("--method-name", type=str, default="None", 
                        choices=['weight_averaging', 'task_arithmetic', 'tw_adamerging', 'lw_adamerging', 'ties_merging', 'tw_adamergingpp', 'lw_adamergingpp'])
    
    parser.add_argument("--model-name", type=str, default="ViT-B-32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--one", type=bool, default=False)

    parsed_args = parser.parse_args()

    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return parsed_args
    
    
if __name__ == '__main__':
    args = get_args()
    set_seed(args.seed)

    exam_datasets =  ['SUN397', 'Cars', 'RESISC45', 'SVHN', 'DTD', 'MNIST', 'GTSRB', 'EuroSAT'] 
    learn_datasets = ['SUN397', 'Cars', 'RESISC45', 'SVHN', 'DTD', 'MNIST', 'GTSRB', 'EuroSAT']   
    # ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD', 'ImageNet100', 'CIFAR100', "Homeoffice_Real_World"]
    iterations = 100

    args.model = args.model_name
    args.data_location = dataset_path
    args.save = checkpoint_path + args.model_name
    args.output_path = output_path

    args.logs_path = args.output_path +  'merge_only/' + args.model_name
    args.base_checkpoint = checkpoint_path + args.model_name + '/zeroshot.pt'
    log = create_log_dir(args.logs_path, 'merge_wo_postTraining.txt')

        
    # =============== start merge ===================
    base_model, alpha_model = get_merged_model()

    log.info(f"================= (merge w/o post_train) eval performance of {args.method_name} ===============")

    Total_ACC = 0.
    ACC_per_Dataset = []
    Dataset_names = []
    for dataset_name in learn_datasets:
        image_encoder = alpha_model.get_image_encoder()
        classification_head = alpha_model.get_classification_head(dataset_name)
        metrics = eval_single_dataset_preprocess_head(image_encoder, classification_head, dataset_name, args)
        test_acc = metrics['top1']
    
        Total_ACC += test_acc
        ACC_per_Dataset.append(metrics['top1'])
        Dataset_names.append(dataset_name)

    Val_ACC = Total_ACC / len(learn_datasets)
    log.info("=== " + ''.join(f"{s:<10}" for s in Dataset_names) + "===")  # dataset name    
    log.info("=== " + ''.join(f"{num:<10.4f}" for num in ACC_per_Dataset) + "===")  # the corresponding val_acc
    log.info(f'=== Avg acc: {str(Val_ACC)[:6]}')