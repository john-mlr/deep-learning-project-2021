#!/bin/bash

#BSUB -J "ptnoattention"
#BSUB -P acc_tauomics
#BSUB -q gpu
#BSUB -R a100
#BSUB -R "rusage[ngpus_excl_p=1]"
#BSUB -R rusage[mem=8000]
#BSUB -R "select[hname!=lg07c07 && hname!=lg07c08]"
#BSUB -n 8
#BSUB -W 144:00
#BSUB -oo /sc/arion/projects/tauomics/BYOL-tau-scoring/attention/ptnoattention/output.txt
#BSUB -eo /sc/arion/projects/tauomics/BYOL-tau-scoring/attention/ptnoattention/error.txt

cd /sc/arion/projects/tauomics/BYOL-tau-scoring/attention

ml purge
ml proxies
ml anaconda3/4.6.4
ml cuda/11.1
ml git

source activate Torch_DL

python ./braak_scoring_attention.py -n 1 -g 1 -nr 0 -e 1000 -b 25 \
--dump_path ./ptnoattention \
--split 0.75 \
