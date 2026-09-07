# Linux CUDA runner operations

The `Linux CUDA` workflow runs only on pushes to `dev` and `master`, manual
dispatch, and the default branch's nightly schedule. Pull-request builds use
GitHub-hosted machines. A runner must have the `self-hosted`, `linux`, and
`cuda` labels, CMake 3.24+, Ninja, a C++20 compiler, CUDA, cuDNN 9, and a working
NVIDIA GPU. The job checks `nvidia-smi` and `nvcc` before full verification.

Configure the intended CUDA compiler through the runner's `PATH`. Nonstandard
cuDNN installations also require `OPENNN_CUDNN_INCLUDE_DIR` and
`OPENNN_CUDNN_LIBRARY`; `OPENNN_CUDA_ARCHITECTURES` can set the GPU architecture.
These paths belong in the runner environment, not in the repository workflow.

The runner `opennn-wsl-cuda` was registered on 2026-09-07. It uses a dedicated
Linux account, `opennn-ci`, and a systemd service. Its service restricts writes
to the runner directory and its own home, hides other Linux home directories
and the Windows C: mount, and prohibits gaining privileges. WSL DNS remains
accessible. This is a workstation runner; it can accept jobs only while its
Windows host is awake, online, and WSL is running.

On that workstation, the Windows scheduled task `OpenNN Linux CUDA runner`
starts a hidden WSL keepalive at sign-in. The Linux service starts with WSL.
The task does not wake the computer or make the runner available while it is
powered off. A dedicated always-on GPU host would remove this limitation.

Service operations in WSL:

```bash
sudo systemctl status actions.runner.Artelnics-opennn.opennn-wsl-cuda.service
sudo journalctl -u actions.runner.Artelnics-opennn.opennn-wsl-cuda.service -n 50
sudo systemctl restart actions.runner.Artelnics-opennn.opennn-wsl-cuda.service
```

To pause this runner, stop its Linux service. To disable startup permanently,
disable the service and the Windows scheduled task. Remove its registration in
GitHub repository Settings > Actions > Runners if retiring the host.

The implementation follows GitHub's [runner registration](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/add-runners)
and [Linux service](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/configure-the-application)
documentation. Keep this runner off untrusted pull-request workflows; public
repository code must be reviewed before it runs on a persistent host.
