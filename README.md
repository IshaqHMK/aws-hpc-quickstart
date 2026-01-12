# AWS HPC (AUS) Quickstart

This repository is a beginner guide to access the American University of Sharjah High Performance Computing (HPC) environment from a Windows laptop, move files to the cluster, and run GPU or CPU jobs using the Slurm scheduler. HPC is a set of powerful computers (many CPU cores, large RAM, and GPUs) designed to run heavy workloads faster than a normal laptop.

## Why you might need it
AUS HPC is useful when a laptop becomes too slow or too limited for tasks such as:
- neural network training (especially on GPUs)
- large simulations
- parameter sweeps and batch experiments
- long running computations that should not depend on keeping your laptop on

## How it works at AUS (high level)
AUS HPC has a few key pieces:
- VPN (AWS Client VPN): secure tunnel into AUS network when off campus
- Login nodes (SSH or DCV): where you connect, edit files, and submit jobs
- Shared storage (`/shared/<username>`): where your code and data should live so compute nodes can access them
- Compute nodes (CPU and GPU): where the actual job runs
- Slurm scheduler: the system that queues jobs and allocates compute resources

Important: SSH or DCV gets you into the login node. The real training or computation runs on compute nodes after submitting a job with Slurm.

## What you get
- Secure access using AWS Client VPN plus AUS SSO
- SSH access to the HPC login node
- Shared storage under `/shared/<username>/...`
- Job submission using Slurm (`sbatch`, `squeue`, `sinfo`, `sacctmgr`)
- Optional GUI desktop using Amazon DCV (with VS Code inside DCV)
- Drag and drop file transfer from Windows using WinSCP (SFTP)

## Prerequisites
- AUS account username and password
- AUS provided VPN configuration file (downloaded from AUS SSO portal or the AUS HPC guide page)
- Off campus access requires VPN

## Install on Windows (local laptop)

### 1) AWS Client VPN
Download:
- https://aws.amazon.com/vpn/client-vpn-download/
- https://self-service.clientvpn.amazonaws.com/endpoints/cvpn-endpoint-05fd3a7ddfd1494c4

Windows usage guide:
- https://docs.aws.amazon.com/vpn/latest/clientvpn-user/client-vpn-connect-windows.html

### 2) Amazon DCV client (optional, for GUI desktop)
Download:
- https://www.amazondcv.com/

Client documentation:
- https://docs.aws.amazon.com/dcv/latest/userguide/client.html

### 3) WinSCP (recommended for easy copy from Windows to HPC)
Download:
- https://winscp.net/eng/download.php

### 4) VS Code Remote SSH (optional alternative workflow)
Extension:
- https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh

Docs:
- https://code.visualstudio.com/docs/remote/ssh

## Connection overview (what happens every time)
1. Connect VPN (required off campus)
2. SSH into a login node (or connect using DCV for GUI)
3. Work inside `/shared/<username>/...`
4. Submit jobs with Slurm (`sbatch`)
5. Monitor jobs (`squeue`) and view output logs

## Step 1: Connect VPN (AWS Client VPN)
1. Open AWS Client VPN.
2. Import the AUS VPN configuration file (.ovpn or similar) if not already added.
3. Click Connect.
4. A browser opens AUS SSO. Log in with AUS credentials.
5. Confirm AWS Client VPN status shows Connected.

## Step 2: SSH to a login node

Prereq: AWS Client VPN must show Connected.

### CPU login
On your Windows laptop, open PowerShell and run:
```bash
ssh <username>@hpc-login-gen.aus.edu
```

### GPU login

```bash
ssh <username>@hpc-login-gpu.aus.edu
```

First time you will see a host key prompt. Type `yes`.

You know you are on the HPC when the prompt looks like this:

```text
[ihafez@ip-10-240-16-55 ~]$
```

Important:

* If you see `PS C:\...>` you are on your laptop, not HPC.
* Slurm commands work only after you are inside the HPC SSH session.

## Step 3: Create your working folder on shared storage

Run on the HPC terminal:

```bash
mkdir -p /shared/$USER/nn_training
cd /shared/$USER/nn_training
pwd
```

Expected:

```text
/shared/<username>/nn_training
```

## Step 4: Get your Slurm account and partitions

Account:

```bash
sacctmgr show assoc user=$USER format=User,Account
```

Example output we got:

```text
User     Account
ihafez   acc-rdhao+
```

Partitions:

```bash
sinfo
```

Example output we got:

```text
PARTITION AVAIL TIMELIMIT NODES STATE NODELIST
cpu*      up    infinite  60    idle~ cpu-dy-c5-0-[1-60]
gpu       up    infinite  30    idle~ gpu-dy-g5-0-[1-30]
hpc       up    infinite  30    idle~ hpc-dy-h6-0-[1-30]
```

Correct values for you:

* Account: `acc-rdhaouadi`
* GPU partition exists: `gpu`

If your account ever prints weird characters like a trailing `+`, do not trust it. Re-run the command and use the clean account name (for you it is `acc-rdhaouadi`).

## Step 5: Create a Python environment in /shared (Python 3.10)

We did this because the default python was Python 3.13 and torch was missing.

Run:

```bash
mkdir -p /shared/$USER/conda_envs
source /opt/miniconda/etc/profile.d/conda.sh
```

If conda complains about Terms of Service, accept them:

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

Now create the environment:

```bash
conda create -y -p /shared/$USER/conda_envs/torch310 python=3.10
conda activate /shared/$USER/conda_envs/torch310
python -V
```

Expected:

```text
Python 3.10.x
```

## Step 6: Install PyTorch correctly (what worked)

Conda install failed because of glibc constraints, especially with torchvision. What worked was installing torch using pip inside the env.

Inside the env:

```bash
python -m pip install -U pip setuptools wheel
pip install torch
pip install numpy scipy pandas matplotlib tqdm tensorboard pyyaml scikit-learn torchinfo einops
```

Quick import test:

```bash
python -c "import torch, numpy; print(torch.__version__); print(numpy.__version__)"
```

Note: CUDA availability will show False on the login node because it has no GPU. That is normal.

## Step 7: Create the Slurm job script correctly (sbatch file)

Critical rules:

1. The first line must be `#!/bin/bash` with no spaces before it.
2. The file must not contain Windows line endings.
3. Use your correct account and QoS.

From `/shared/$USER/nn_training`, create or overwrite the sbatch file:

```bash
cat > train_gpu.sbatch <<'EOF'
#!/bin/bash
#SBATCH --account=acc-rdhaouadi
#SBATCH --partition=gpu
#SBATCH --qos=gpu-long-rdhaouadi-001
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --job-name=nn_test
#SBATCH --output=slurm_%j.out

cd /shared/$USER/nn_training
source /opt/miniconda/etc/profile.d/conda.sh
conda activate /shared/$USER/conda_envs/torch310
python train.py
EOF
```

Then clean line endings:

```bash
dos2unix train_gpu.sbatch 2>/dev/null || sed -i 's/\r$//' train_gpu.sbatch
```

Verify the first lines:

```bash
head -n 5 train_gpu.sbatch
cat -A train_gpu.sbatch | head -n 3
```

You must not see `^M`.

## Step 8: Create a GPU sanity check python script

Create `train.py` in the same folder:

```bash
cat > train.py <<'EOF'
import torch, time

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = torch.randn(30000, 512, device=device)
w = torch.randn(512, 1, device=device, requires_grad=True)

t0 = time.time()
for epoch in range(5):
    y = x @ w
    loss = (y**2).mean()
    loss.backward()
    with torch.no_grad():
        w -= 1e-3 * w.grad
        w.grad.zero_()
    print("epoch", epoch, "loss", float(loss))

print("elapsed_sec", time.time() - t0)
EOF
```

## Step 9: Submit, monitor, and read output

Submit:

```bash
sbatch train_gpu.sbatch
```

It returns a job id like:

```text
Submitted batch job 117530
```

Monitor:

```bash
squeue -u $USER
```

Common states:

* `CF` configuring, normal right after submit
* `R` running
* disappears from queue when finished

Check final result:

```bash
sacct -j <jobid> --format=JobID,State,Elapsed,ExitCode
```

Read output:

```bash
ls -lh slurm_*.out
less slurm_<jobid>.out
```

Exit less with `q`.

Your success criteria in the output:

* `cuda available: True`
* GPU name printed (A10G on your cluster)

## Step 10: Copy your real code and data from your laptop to HPC

Windows File Explorer cannot see `/shared/...` because it is on the remote HPC.

Use WinSCP:

* Protocol: SFTP
* Host: `hpc-login-gpu.aus.edu`
* Port: `22`
* Username: your AUS username
* Password: AUS password

Remote folder to drop files into:

```text
/shared/<username>/nn_training
```

After copying, verify on HPC:

```bash
cd /shared/$USER/nn_training
ls -lh
```

## Amazon DCV desktop (GUI) plus VS Code inside DCV (optional)

### 1) Create a DCV session

If DCV says there is no session, SSH into the login node once. The system may auto create a session.

### 2) Connect using DCV client

In Amazon DCV client:

* Host: `hpc-login-gpu.aus.edu` (or the login node you used)
* Username: your AUS username
* Password: your AUS password

### 3) Open the shared folder in the DCV file manager

Within the DCV desktop, use the Linux "Files" app and browse:
`/shared/<username>/nn_training`

Tip: you can create a shortcut:

```bash
ln -sfn /shared/$USER/nn_training ~/nn_training
```

Then open `~/nn_training` from the DCV file manager.

### 4) Open VS Code inside DCV

In a DCV terminal:

```bash
code /shared/$USER/nn_training
```

If `code` runs but nothing obvious happens, check if VS Code opened behind other windows.

## Copy files from Windows to HPC using WinSCP (what we used)

### 1) Keep VPN connected

AWS Client VPN must show Connected.

### 2) WinSCP new connection

* File protocol: SFTP
* Host name: `hpc-login-gpu.aus.edu`
* Port number: `22`
* User name: your AUS username
* Password: your AUS password

### 3) Drag and drop

Remote target folder:
`/shared/<username>/nn_training`

Drag files or whole folders from the Windows pane to the remote pane.

### 4) Verify on HPC

In SSH:

```bash
cd /shared/$USER/nn_training
ls -lh
```

