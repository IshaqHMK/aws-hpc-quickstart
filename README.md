# AUS HPC quickstart (VPN, SSH, DCV, Slurm, file copy)

This README documents the exact workflow we used to access the AUS HPC cluster, create a shared project folder, submit a GPU Slurm job, and copy files from a Windows laptop to the cluster.

## What you get
- Secure access using AWS Client VPN plus AUS SSO
- SSH access to the HPC login node
- Shared storage under `/shared/<username>/...`
- Job submission using Slurm (sbatch, squeue, sinfo, sacctmgr)
- Optional GUI desktop using Amazon DCV (with VS Code inside DCV)
- Drag and drop file transfer from Windows using WinSCP (SFTP)

## Prerequisites
- AUS account username and password
- AUS provided VPN configuration file (downloaded from AUS SSO portal or the AUS HPC guide page)
- Off campus access requires VPN

## Install on Windows (local laptop)

### 1) AWS Client VPN
Download and install:
https://aws.amazon.com/vpn/client-vpn-download/

AWS docs (Windows):
https://docs.aws.amazon.com/vpn/latest/clientvpn-user/client-vpn-connect-windows.html

### 2) Amazon DCV client (optional, for GUI desktop)
Download:
https://www.amazondcv.com/

AWS docs (client overview):
https://docs.aws.amazon.com/dcv/latest/userguide/client.html

### 3) WinSCP (recommended for easy copy from Windows to HPC)
Download:
https://winscp.net/eng/download.php

### 4) VS Code Remote SSH (optional alternative workflow)
Extension:
https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh

VS Code docs:
https://code.visualstudio.com/docs/remote/ssh

## Connect to AUS HPC

### Step 1) Connect VPN (AWS Client VPN)
1. Open AWS Client VPN.
2. Import AUS VPN configuration file (.ovpn or similar).
3. Click Connect.
4. Browser opens AUS SSO, log in with your AUS credentials.
5. Confirm AWS Client VPN status shows Connected.

### Step 2) SSH to the login node
Use a terminal on Windows (PowerShell works):

CPU login:
```bash
ssh <username>@hpc-login-gen.aus.edu
````

GPU login:

```bash
ssh <username>@hpc-login-gpu.aus.edu
```

First time you will see a host key prompt. Type `yes`.

Important note:
If you run Slurm commands on Windows PowerShell, they will fail.
Slurm commands work only after you are inside the HPC SSH session.

## Create your shared project folder (on the HPC, inside SSH)

After SSH login, run:

```bash
mkdir -p /shared/$USER/nn_training
cd /shared/$USER/nn_training
pwd
```

Expected:
`/shared/<username>/nn_training`

## Find your Slurm account and partitions (on the HPC)

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

## Run a GPU example job (PyTorch) using Slurm

### 1) Create `train.py` (on the HPC)

Inside `/shared/$USER/nn_training`:

```bash
nano train.py
```

Paste:

```python
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(20000, 256, device=device)
w = torch.randn(256, 1, device=device, requires_grad=True)

for epoch in range(10):
    y = x @ w
    loss = (y**2).mean()
    loss.backward()
    with torch.no_grad():
        w -= 1e-3 * w.grad
        w.grad.zero_()
    print("epoch", epoch, "loss", float(loss))

print("done")
```

Save and exit nano.

### 2) Create `train_gpu.sbatch` (on the HPC)

```bash
nano train_gpu.sbatch
```

Paste, and replace `YOUR_ACCOUNT` with your Slurm account (example: `acc-rdhao+`):

```bash
#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --job-name=nn_test
#SBATCH --output=slurm_%j.out

cd /shared/$USER/nn_training
python3 train.py
```

### 3) Submit the job (on the HPC)

```bash
cd /shared/$USER/nn_training
sbatch train_gpu.sbatch
```

You will get a job id:
`Submitted batch job <jobid>`

### 4) Monitor (on the HPC)

```bash
squeue -u $USER
```

### 5) View output (on the HPC)

After the job finishes:

```bash
ls -lh
less slurm_*.out
```

You want to see:

* `cuda available: True`
* epoch loss values

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

Tip:
If the file chooser is confusing, you can create a shortcut:

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

## Copy files from Windows to HPC using WinSCP (this is what we used)

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

## Common mistakes and fixes

### Mistake: running `sacctmgr` or `sinfo` on Windows PowerShell

Symptom: "command not recognized"
Fix: SSH into the HPC first. Run Slurm commands only inside the HPC SSH session.

### Mistake: expecting Windows File Explorer to show `/shared/...`

Fix: `/shared/...` is on the Linux HPC.
Use WinSCP (SFTP) for file copy, or use VS Code Remote SSH to edit on the HPC directly.

## Suggested repo structure (example)

* `scripts/` for sbatch files and helpers
* `src/` for training code
* `data/` for small sample data (do not commit big datasets)
* `results/` for logs and outputs (often not committed)

Example:

```text
nn_training/
  README.md
  train.py
  train_gpu.sbatch
  scripts/
  src/
  results/
```

```
::contentReference[oaicite:0]{index=0}
```
