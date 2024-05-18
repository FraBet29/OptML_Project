# How to access to the cluster

To connect to the clusters, you have to be inside the EPFL network or [establish a VPN connection](https://www.epfl.ch/campus/services/en/it-services/network-services/remote-intranet-access/vpn-clients-available/) [[link](https://scitas-doc.epfl.ch/user-guide/using-clusters/connecting-to-the-clusters/)].

## Login

You can access the clusters by using `ssh` on your own computer. The command is:

```bash
ssh -X [username]@izar.epfl.ch
```

For example:

```bash
ssh -X milikic@izar.epfl.ch
```

Or better yet, you can use VSCode's SSH extension to connect to the cluster. You can find the instructions [here](https://code.visualstudio.com/docs/remote/ssh). In short, you have to install the extension, click on the green button in the bottom left corner, and select `Remote-SSH: Connect to Host...`, followed by `Add New SSH Host...`. Then, you can enter the code from above.

Login password is the same as your EPFL password.

## Create conda environment

**ONLY FOR THE FIRST TIME** you have to download and install miniconda and set up the environment. You can do this by running the following script:

```bash
./install.sh
```

Now if needed, you can activate the environment by running:

```bash
conda activate dl
```

Don't forget to intall other dependencies!

## Submitting a job

To submit a job, you have to run `job.run` with the following command:

```bash
sbatch job.run
```

You will obtain a job ID as an output. You can check the status and id of your jobs by running:

```bash
squeue -u [username]
```

If job is running, the status (ST column) will be `R`. If it is pending, the status will be `PD`. Because the cluster is shared, you might have to wait for your job to start. Once the job has started, you can connect to the node by running:

```bash
srun --pty --jobid <JOBID> /bin/bash
```

This will open a new terminal on the node where your job is running and where GPU is available. You can check the GPU usage by running:

```bash
nvidia-smi
```

I set up the job to be active for 24 hours. If you need more time, you can change the `--time` parameter in `job.run` file. No need to start a new job, before the current one finishes.

To kill a job you can run:

```bash
scancel <job-id>
```

## Start the jupyter notebook server in the compute node

1. Login the compute node on the clusters. Replace `<JOBID>` with the `JOBID` you noted down in the previous step.
   ```bash
   srun --pty --jobid <JOBID> /bin/bash
   ```
2. Start the jupyter notebook server.

   ```bash
   jupyter-notebook --no-browser --port=8888
   ```

   Please note the line with the token `<token>` and the IP address `<ip-address>`. You will need it in the next step.

   For example, the line looks like

   ```
   http://10.91.24.11:8888/tree?token=c016e3b2e400c8f44162428200ba2df017b3393581916c5d
   ```

   where the ip address `<ip-address>` is `10.91.24.11` and the token `<token>` is `c016e3b2e400c8f44162428200ba2df017b3393581916c5d`.

3. Forward the port from the compute node to your local machine, by executing the following command on **your local machine**. Replace `<ip-address>` with the IP address you noted down in the previous step.
   ```bash
   ssh -L 8888:<ip-address>:8888 -l [username] izar.epfl.ch -f -N
   ```
4. On your local machine, open a web browser and enter the following URL. Replace `<token>` with the token you noted down in the previous step.

   ```
   http://127.0.0.1:8888/tree?token=<token>
   ```

5. In case it does not work, you can try to use the following command on your local machine.

   ```bash
   ssh -t -t [username]@izar.epfl.ch -L 8888:localhost:8888 ssh i31 -L 8888:localhost:8888
   ```
