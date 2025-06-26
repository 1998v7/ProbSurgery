import os
import sys
from path_config import *
sys.path.append(src_root_path)
import torch
import argparse
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

from src.utils import create_log_dir, set_seed
from src.eval import eval_single_dataset
from src.ties_merging_utils import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-location", type=str, default=os.path.expanduser('~/data'), help="The root directory for the datasets.",)
    parser.add_argument("--eval-datasets", default=None, type=lambda x: x.split(","), help="Which datasets to use for evaluation. Split by comma, e.g. MNIST,EuroSAT. ")
    parser.add_argument("--train-dataset", default=None, type=lambda x: x.split(","), help="Which dataset(s) to patch on.",)
    parser.add_argument("--batch-size",type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--openclip-cachedir", type=str,default='./.cache/open_clip',help='Directory for caching models from OpenCLIP')
    
    parser.add_argument("--method-name", type=str, default="weight_averaging", 
                        choices=['weight_averaging', 'task_arithmetic', 'tw_adamerging', 'lw_adamerging', 'ties_merging', 'tw_adamergingpp', 'lw_adamergingpp'])
    
    parser.add_argument("--model-name", type=str, default="ViT-B-32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--KL_weight", type=float, default=1e-4)
    parsed_args = parser.parse_args()

    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return parsed_args
    
    
if __name__ == '__main__':
    args = get_args()
    set_seed(args.seed)

    exam_datasets =  ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD']  

    iterations = 100
    eval_iterations = 100

    args.model = args.model_name
    args.data_location = dataset_path
    args.save = checkpoint_path + args.model_name

    args.logs_path = 'logs/' + args.model_name
    args.base_checkpoint = checkpoint_path + args.model_name + '/zeroshot.pt'
    log = create_log_dir(args.logs_path, 'zero_shot.txt')

    log.info(f"================= Zero_shot performance on {exam_datasets} ==================")
    base_model = torch.load(args.base_checkpoint)
    Total_ACC = 0.
    for dataset_name in exam_datasets:
        metrics = eval_single_dataset(base_model, dataset_name, args)
        Total_ACC += metrics['top1']
        log.info('Eval on dataset: ' + str(dataset_name) + ' ACC: ' + str(metrics['top1']))
        
    val_acc = Total_ACC / len(exam_datasets)
    log.info('(zero-shot performance) Eval: Average acc is: ' + str(val_acc) + '\n')