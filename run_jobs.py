import os
from time import sleep

def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)


job_directory = "%s/.job" % os.getcwd()
scratch = os.environ['SCRATCH']

# Make top level directories
mkdir_p(job_directory)

study = 'her_study'
nb_seeds = 2
replay_k_list = [1, 2, 3, 4]
strategy_list = ['final', 'future']

for i in range(nb_seeds):
    for replay_k in replay_k_list:
        for strategy in strategy_list:
            job_file = os.path.join(job_directory, f"{study}_{strategy}_{replay_k}%.slurm")

            with open(job_file, 'w') as fh:
                fh.writelines("#!/bin/bash\n")
                fh.writelines("#SBATCH --account=kcr@v100\n")
                fh.writelines(f"#SBATCH --job-name={study}_{strategy}_{replay_k}\n")
                fh.writelines("#SBATCH --qos=qos_gpu-t3\n")
                fh.writelines(f"#SBATCH --output={study}_{strategy}_{replay_k}%_%j.out\n")
                fh.writelines(f"#SBATCH --error={study}_{strategy}_{replay_k}%_%j.out\n")
                fh.writelines("#SBATCH --time=19:59:59\n")
                fh.writelines("#SBATCH --ntasks=24\n")
                fh.writelines("#SBATCH --ntasks-per-node=1\n")
                fh.writelines("#SBATCH --gres=gpu:1\n")
                fh.writelines("#SBATCH --hint=nomultithread\n")
                fh.writelines("#SBATCH --array=0-0\n")

                fh.writelines("module load pytorch-gpu/py3/1.4.0\n")

                fh.writelines("export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gpfslocalsup/spack_soft/mesa/18.3.6/gcc-9.1.0-bikg6w3g2be2otzrmyy43zddre4jahme/lib\n")
                fh.writelines("export LIBRARY_PATH=$LIBRARY_PATH:/gpfslocalsup/spack_soft/mesa/18.3.6/gcc-9.1.0-bikg6w3g2be2otzrmyy43zddre4jahme/lib\n")
                fh.writelines("export CPATH=$CPATH:/gpfslocalsup/spack_soft/mesa/18.3.6/gcc-9.1.0-bikg6w3g2be2otzrmyy43zddre4jahme/include\n")
                fh.writelines("export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/linkhome/rech/genisi01/uqy56ga/.mujoco/mujoco200/bin\n")
                fh.writelines("export OMPI_MCA_opal_warn_on_missing_libcuda=0\n")
                fh.writelines("export OMPI_MCA_btl_openib_allow_ib=1\n")
                fh.writelines("export OMPI_MCA_btl_openib_warn_default_gid_prefix=0\n")
                fh.writelines("export OMPI_MCA_mpi_warn_on_fork=0\n")

                fh.writelines(f"srun python -u -B train.py --algo 'semantic' --n-blocks 5 --n-epochs 1000 --n-cycles 50 --n-batches 30 --architecture 'relation_network' --save-dir '{study}_{strategy}_{replay_k}/' 2>&1 ")

            os.system("sbatch %s" % job_file)
            sleep(1)
        