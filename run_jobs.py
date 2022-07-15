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

betas = [20, 50, 100, 200, 500]
planning_probas = [0]
nb_seeds = 3

for s in range(nb_seeds):
    for beta in betas:
        job_file = os.path.join(job_directory, f"main_beta={beta}%.slurm")

        with open(job_file, 'w') as fh:
            fh.writelines("#!/bin/bash\n")
            fh.writelines("#SBATCH --account=kcr@v100\n")
            fh.writelines(f"#SBATCH --job-name=main_beta={beta}\n")
            fh.writelines("#SBATCH --qos=qos_gpu-t3\n")
            fh.writelines(f"#SBATCH --output=main_beta={beta}_%_%j.out\n")
            fh.writelines(f"#SBATCH --error=main_beta={beta}_%_%j.out\n")
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

            fh.writelines(f"srun python -u -B train.py  --beta {beta} --save-dir 'main_beta={beta}/' 2>&1 ")

        os.system("sbatch %s" % job_file)
        sleep(1)
        