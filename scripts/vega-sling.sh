#!/bin/bash

# NODE_NAME=$(hostname)
# echo ""
# echo "NODE_NAME=$NODE_NAME"
# echo ""

#SBATCH --reservation=zunkovic_2

# #SBATCH --constraint=olimp
# #SBATCH --nodelist=tesla
# #SBATCH --exclude=manycore
# #SBATCH --partition=gridlong
# #SBATCH --gres=${GRES}
# #SBATCH --partition=gpu
# #SBATCH --gres=cuda:1


while IFS='' read -r line || [[ -n "$line" ]]; do
    IFS=';' read -r -a list <<< "$line"
    NAME=${list[0]//[[:blank:]]/}
    PROCESSOR=${list[1]//[[:blank:]]/}
    OMP_THREADS_ALL=${list[2]//[[:blank:]]/}
    OMP_THREADS=${list[3]//[[:blank:]]/}
    TYPE=${list[4]//[[:blank:]]/}
    COMMAND=${list[5]}
   

    GRES=''
    if [ ${PROCESSOR} == 'gpu' ]
    then
    GRES="gpu:1"
    fi
   
    echo "${NAME}, ${PROCESSOR}, ${GRES}, ${OMP_THREADS}, ${OMP_THREADS_ALL}"
   
  JOB=`sbatch <<EOJ
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=${OMP_THREADS_ALL}
#SBATCH --mem=10G
#SBATCH --partition=${PROCESSOR}
#SBATCH --time=2-00:00:00
#SBATCH --job-name=${NAME}
#SBATCH --output=output/${NAME}
#SBATCH --gres=${GRES}

cd /ceph/hpc/home/zegarraa/selfconsistent_xy/bin

module load Python 

ulimit -s unlimited
export OMP_STACKSIZE=4G
export OMP_NUM_THREADS=${OMP_THREADS}
export OPENBLAS_NUM_THREADS=${OMP_THREADS}

srun --ntasks 1 ${COMMAND}

EOJ`
  echo "${JOB}: ${COMMAND}"
done < "$1"
