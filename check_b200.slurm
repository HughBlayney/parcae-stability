#!/bin/bash
# Minimal check that civo-b200-1 is not properly initialised for CUDA.
#
#   sbatch check_b200.slurm     # then: cat slurm_out/checkb200-<jobid>.out
#
# Healthy node:  fabricmanager active,  Fabric State "Completed",  CUDA context OK.
# civo-b200-1 :  fabricmanager FAILED,  Fabric State "In Progress", and ANY real CUDA
#               use (1 GPU or more) dies with "Error 802: system not yet initialized".
#               nvidia-smi and torch.cuda.device_count() still work because neither
#               creates a CUDA context (device_count uses NVML) -- which is why the
#               node looks healthy to Slurm and to a naive check. Only forcing a
#               context (torch.zeros(1).cuda()) surfaces the fault.
#               Fix is to restart nvidia-fabricmanager on the node.
#SBATCH --job-name=checkb200
#SBATCH --account=mlrg_nsg
#SBATCH --partition=b200
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:02:00
#SBATCH --output=./slurm_out/checkb200-%j.out
#SBATCH --error=./slurm_out/checkb200-%j.out

echo "host: $(hostname)"
echo
echo "0) torch.cuda.is_available() -- NOTE: returns False without crashing (error is caught),"
echo "   and device_count() below returns a number via NVML. Neither creates a CUDA context,"
echo "   so these are NOT reliable health checks. Step 4 is the real test."
/users/scro3801/repos/external/parcae/env_parcae/bin/python -c "import torch;print('   is_available ->', torch.cuda.is_available())"
echo
echo "1) nvidia-smi (basic query works even when fabric is broken):"
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
echo
echo "2) nvidia-fabricmanager service:"
systemctl is-active nvidia-fabricmanager
echo
echo "3) NVLink fabric state (should say 'Completed'):"
nvidia-smi -q | grep -A2 -i "Fabric" | grep -i "State"
echo
echo "4) THE REAL TEST -- force a CUDA context on every allocated GPU:"
source /users/scro3801/repos/external/parcae/env_parcae/bin/activate 2>/dev/null
python -c "
import torch
n = torch.cuda.device_count()
for i in range(n):
    torch.zeros(1, device=f'cuda:{i}')   # forces a context; this is what fails on a dead fabric
print('   initialised', n, 'GPU(s) -> RESULT: CUDA OK')
" || echo "   RESULT: CUDA INIT FAILED (see 'Error 802: system not yet initialized' above)"
