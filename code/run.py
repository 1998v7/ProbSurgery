import subprocess


import warnings
warnings.filterwarnings("ignore")

# ======= you can change the following parameter during training =========
# --model-name ViT-B-32 ViT-B-16 ViT-L-14 
# --method-name weight_averaging task_arithmetic ties_merging lw_adamerging
# --loss_func L1_loss mse_Loss smoothL1_Loss CosineSimilarity


Probsurgery_commands =[
    " CUDA_VISIBLE_DEVICES=0 python run/run_ProbSurgery.py --model-name ViT-B-32 --method-name weight_averaging --iter_number 50 & \
        CUDA_VISIBLE_DEVICES=1 python run/run_ProbSurgery.py --model-name ViT-B-32 --method-name task_arithmetic --iter_number 50 & \
        CUDA_VISIBLE_DEVICES=2 python run/run_ProbSurgery.py --model-name ViT-B-32 --method-name ties_merging --iter_number 50 & \
        CUDA_VISIBLE_DEVICES=3 python run/run_ProbSurgery.py --model-name ViT-B-32 --method-name lw_adamerging --iter_number 50",
    
    " CUDA_VISIBLE_DEVICES=0 python run/run_ProbSurgery.py --model-name ViT-B-32 --method-name weight_averaging --iter_number 50 --one True & \
        CUDA_VISIBLE_DEVICES=1 python run/run_ProbSurgery.py --model-name ViT-B-32 --method-name task_arithmetic --iter_number 50 --one True & \
        CUDA_VISIBLE_DEVICES=2 python run/run_ProbSurgery.py --model-name ViT-B-32 --method-name ties_merging --iter_number 50 --one True & \
        CUDA_VISIBLE_DEVICES=3 python run/run_ProbSurgery.py --model-name ViT-B-32 --method-name lw_adamerging --iter_number 50 --one True",
    ]


Surgery_commands =[
    " CUDA_VISIBLE_DEVICES=0 python run/run_Surgery.py --model-name ViT-B-32 --method-name weight_averaging & \
        CUDA_VISIBLE_DEVICES=1 python run/run_Surgery.py --model-name ViT-B-32 --method-name task_arithmetic & \
        CUDA_VISIBLE_DEVICES=2 python run/run_Surgery.py --model-name ViT-B-32 --method-name ties_merging & \
        CUDA_VISIBLE_DEVICES=3 python run/run_Surgery.py --model-name ViT-B-32 --method-name lw_adamerging",
    
    " CUDA_VISIBLE_DEVICES=0 python run/run_Surgery.py --model-name ViT-B-32 --method-name weight_averaging --one True & \
        CUDA_VISIBLE_DEVICES=1 python run/run_Surgery.py --model-name ViT-B-32 --method-name task_arithmetic --one True & \
        CUDA_VISIBLE_DEVICES=2 python run/run_Surgery.py --model-name ViT-B-32 --method-name ties_merging --one True & \
        CUDA_VISIBLE_DEVICES=3 python run/run_Surgery.py --model-name ViT-B-32 --method-name lw_adamerging --one True",
    ]

for cmd in Probsurgery_commands:
    subprocess.run(cmd, shell=True, check=True)
