# Serve Llama2-70B on across 8 Intel Gaudi Accelerator Cards inside a Gaudi2 Node
Deploying the Llama2-70B model across 8 Intel Gaudi accelerator cards within a Gaudi2 node enables efficient and high-performance inference for large language models, using model parallelism.

We will use DeepSpeed. It aids model parallelism in inferencing by efficiently splitting large models across multiple Gaudi2 accelerator cards, enabling the simultaneous use of multiple devices to handle extensive computations. This parallelism reduces memory bottlenecks and accelerates inference by distributing the model's layers or operations, ensuring optimal utilization of available hardware resources.

We will use Ray for serving the model. Using Ray to serve models is helpful because it simplifies scaling and managing distributed model deployments, providing robust support for parallel processing and resource scheduling. Additionally, Ray's seamless integration with various machine learning frameworks and its ability to handle complex workflows ensure efficient and flexible model serving in production environments.

The Intel Gaudi architecture, designed specifically for AI workloads, offers superior computational power and memory bandwidth, ensuring seamless handling of the extensive parameters of Llama2-70B. This setup leverages the Gaudi2 node’s advanced capabilities to optimize throughput and latency, providing a robust solution for applications requiring powerful natural language processing. By integrating Llama2-70B with the Gaudi accelerators, users can achieve scalable and cost-effective AI model serving, making it ideal for enterprise-level deployments.

## Gaudi Node - Starting Point
The Gaudi node in this case we have used has the base setup. Meaning it is setup with the Gaudi software stack.
> Operating System - Ubuntu 22.04
> Intel Gaudi Software Version - 1.16.0

Below is the output from the `lscpu` command:
```
root@denvrbm-1099:/home/ubuntu# lscpu
Architecture:            x86_64
  CPU op-mode(s):        32-bit, 64-bit
  Address sizes:         52 bits physical, 57 bits virtual
  Byte Order:            Little Endian
CPU(s):                  160
  On-line CPU(s) list:   0-159
Vendor ID:               GenuineIntel
  Model name:            Intel(R) Xeon(R) Platinum 8380 CPU @ 2.30GHz
    CPU family:          6
    Model:               106
    Thread(s) per core:  2
    Core(s) per socket:  40
    Socket(s):           2
    Stepping:            6
    CPU max MHz:         3400.0000
    CPU min MHz:         800.0000
    BogoMIPS:            4600.00
    Flags:               fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fx
                         sr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts re
                         p_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx smx
                         est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_t
                         imer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb cat_l3 invpcid_single
                         ssbd mba ibrs ibpb stibp ibrs_enhanced tpr_shadow vnmi flexpriority ept vpid ept_ad fsgsbase ts
                         c_adjust bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma
                          clflushopt clwb intel_pt avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_
                         llc cqm_occup_llc cqm_mbm_total cqm_mbm_local split_lock_detect wbnoinvd dtherm ida arat pln pt
                         s avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg tme avx
                         512_vpopcntdq la57 rdpid fsrm md_clear pconfig flush_l1d arch_capabilities
```

Below is the output from the `hl-smi` command:
```
root@denvrbm-1099:/home/ubuntu# hl-smi
+-----------------------------------------------------------------------------+
| HL-SMI Version:                              hl-1.16.0-fw-50.1.2.0          |
| Driver Version:                                     1.16.0-94aac46          |
|-------------------------------+----------------------+----------------------+
| AIP  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | AIP-Util  Compute M. |
|===============================+======================+======================|
|   0  HL-225              N/A  | 0000:b3:00.0     N/A |                   0  |
| N/A   26C   N/A    82W / 600W |    768MiB / 98304MiB |     0%           N/A |
|-------------------------------+----------------------+----------------------+
|   1  HL-225              N/A  | 0000:b4:00.0     N/A |                   0  |
| N/A   26C   N/A    96W / 600W |    768MiB / 98304MiB |     0%           N/A |
|-------------------------------+----------------------+----------------------+
|   2  HL-225              N/A  | 0000:cc:00.0     N/A |                   0  |
| N/A   27C   N/A    71W / 600W |    768MiB / 98304MiB |     0%           N/A |
|-------------------------------+----------------------+----------------------+
|   3  HL-225              N/A  | 0000:cd:00.0     N/A |                   0  |
| N/A   26C   N/A    89W / 600W |    768MiB / 98304MiB |     0%           N/A |
|-------------------------------+----------------------+----------------------+
|   4  HL-225              N/A  | 0000:19:00.0     N/A |                   0  |
| N/A   26C   N/A    91W / 600W |    768MiB / 98304MiB |     0%           N/A |
|-------------------------------+----------------------+----------------------+
|   5  HL-225              N/A  | 0000:1a:00.0     N/A |                   0  |
| N/A   31C   N/A   100W / 600W |    768MiB / 98304MiB |     0%           N/A |
|-------------------------------+----------------------+----------------------+
|   6  HL-225              N/A  | 0000:43:00.0     N/A |                   0  |
| N/A   26C   N/A   101W / 600W |  27098MiB / 98304MiB |     0%           N/A |
|-------------------------------+----------------------+----------------------+
|   7  HL-225              N/A  | 0000:44:00.0     N/A |                   0  |
| N/A   25C   N/A    99W / 600W |    768MiB / 98304MiB |     0%           N/A |
|-------------------------------+----------------------+----------------------+
| Compute Processes:                                               AIP Memory |
|  AIP       PID   Type   Process name                             Usage      |
|=============================================================================|
|   0        N/A   N/A    N/A                                      N/A        |
|   1        N/A   N/A    N/A                                      N/A        |
|   2        N/A   N/A    N/A                                      N/A        |
|   3        N/A   N/A    N/A                                      N/A        |
|   4        N/A   N/A    N/A                                      N/A        |
|   5        N/A   N/A    N/A                                      N/A        |
|   6       618796     C   ray::ServeRepli                         26330MiB
|   7        N/A   N/A    N/A                                      N/A        |
+=============================================================================+
```
## Gaudi Software and PyTorch Environment Setup
Set Up Intel Gaudi Software Stack. Since our Gaudi software version is 1.16.0, we will be followiing this link - [https://docs.habana.ai/en/v1.16.0/Installation_Guide/Bare_Metal_Fresh_OS.html](https://docs.habana.ai/en/v1.16.0/Installation_Guide/Bare_Metal_Fresh_OS.html).

The support matrix for Gaudi software version 1.16.0 is here - [https://docs.habana.ai/en/v1.16.0/Support_Matrix/Support_Matrix.html](https://docs.habana.ai/en/v1.16.0/Support_Matrix/Support_Matrix.html)

The Gaudi Software Stack verification can be performed using information in the link here - [https://docs.habana.ai/en/v1.16.0/Installation_Guide/SW_Verification.html](https://docs.habana.ai/en/v1.16.0/Installation_Guide/SW_Verification.html)

Please use the correct instructions and support matrix based on your Gaudi Software version.
Run the following instructions, you should be logged into the Gaudi node and be present at the folder level `/home/ubuntu`. This is for a machine with Ubuntu OS and user `ubuntu' logged in.

```
wget -nv https://vault.habana.ai/artifactory/gaudi-installer/1.16.0/habanalabs-installer.sh
```

```
chmod +x habanalabs-installer.sh
```
```
./habanalabs-installer.sh install --type base
```

After running the baove instructions, run the command below.
```
apt list --installed | grep habana
```

You will see an output like below.
```
+=============================================================================+
root@denvrbm-1099:/home/ubuntu# apt list --installed | grep habana

WARNING: apt does not have a stable CLI interface. Use with caution in scripts.

habanalabs-container-runtime/jammy,now 1.16.1-7 amd64 [installed]
habanalabs-dkms/jammy,now 1.16.0-526 all [installed,upgradable to: 1.16.1-7]
habanalabs-firmware-tools/jammy,now 1.16.0-526 amd64 [installed,upgradable to: 1.16.1-7]
habanalabs-firmware/jammy,now 1.16.0-526 amd64 [installed,upgradable to: 1.16.1-7]
habanalabs-graph/jammy,now 1.16.0-526 amd64 [installed,upgradable to: 1.16.1-7]
habanalabs-qual/jammy,now 1.16.0-526 amd64 [installed,upgradable to: 1.16.1-7]
habanalabs-rdma-core/jammy,now 1.16.0-526 all [installed,upgradable to: 1.16.1-7]
habanalabs-thunk/jammy,now 1.16.0-526 all [installed,upgradable to: 1.16.1-7]
```
Run the command.
```
./habanalabs-installer.sh install -t dependencies
```

```
./habanalabs-installer.sh install --type pytorch --venv
```

```
source habanalabs-venv/bin/activate
```
Run the command below.
```
pip list | grep habana
```

You will see an output like this.
```
(habanalabs-venv) ubuntu@denvrbm-1099:~$
(habanalabs-venv) ubuntu@denvrbm-1099:~$ pip list | grep habana
habana_gpu_migration        1.16.0.526
habana-media-loader         1.16.0.526
habana-pyhlml               1.16.1.7
habana_quantization_toolkit 1.16.0.526
habana-torch-dataloader     1.16.0.526
habana-torch-plugin         1.16.0.526
lightning-habana            1.5.0
optimum-habana              1.11.1
```
This shows all the components are installed properly. Run the command below to come out of the virtual environment. We will relaunch the virtual environment again afterwards.

Exit the virtual environment.
```
deactivate
```

## Install Docker
We will install docker on this machine, we will use instructions for Ubuntu 22.04. Recommend running one command at a time.

```
sudo apt update

sudo apt install -y apt-transport-https ca-certificates curl software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update

apt-cache policy docker-ce

sudo apt install -y docker-ce

sudo systemctl status docker
```
The last command shows Docker service running. Press `Control + C` on a Windows keyboard to break out of the docker status and come back to the command prompt.

## Instal Habana Container Runtime
This will setup the Habana container runtime on the Gaudi2 node. This is install for the bare metal installation type. The documentation is available here - [https://docs.habana.ai/en/latest/Installation_Guide/Bare_Metal_Fresh_OS.html](https://docs.habana.ai/en/latest/Installation_Guide/Bare_Metal_Fresh_OS.html).

Run the instructions below.

1. Download and install the public key:

```
curl -X GET https://vault.habana.ai/artifactory/api/gpg/key/public | sudo apt-key add --
```

2. Get the name of the operating system:

```
lsb_release -c | awk '{print $2}'
```
In our case, it shows:
```
ubuntu@denvrbm-1099:~$ lsb_release -c | awk '{print $2}'
jammy
```

3. Create an apt source file /etc/apt/sources.list.d/artifactory.list with deb https://vault.habana.ai/artifactory/debian <OS name from previous step> main content.

```
sudo nano /etc/apt/sources.list.d/artifactory.list
```
Paste in the line `deb https://vault.habana.ai/artifactory/debian jammy main`

Press `Control` + `x` on Windows keyboard, then type `Y` and then press `Enter` to confirm and save the file

In our case, the file at `/etc/apt/sources.list.d/artifactory.list` shows:
```
ubuntu@denvrbm-1099:~$
ubuntu@denvrbm-1099:~$ cat /etc/apt/sources.list.d/artifactory.list
deb https://vault.habana.ai/artifactory/debian jammy main
```

4. Update Debian cache:

```
sudo dpkg --configure -a

sudo apt-get update
```
**Firmware Installation:**
To install the FW, run the following:
```
sudo apt install -y --allow-change-held-packages habanalabs-firmware
```
**Driver Installation:**
1. Run the below command to install all drivers:

```
sudo apt install -y --allow-change-held-packages habanalabs-dkms
```
2. Unload the drivers in this order - habanalabs, habanalabs_cn, habanalabs_en and habanalabs_ib:
**Might take a few mins, please be patient**

```
sudo modprobe -r habanalabs
sudo modprobe -r habanalabs_cn
sudo modprobe -r habanalabs_en
sudo modprobe -r habanalabs_ib
```
3. Load the drivers in this order - habanalabs_en and habanalabs_ib, habanalabs_cn, habanalabs:

```
sudo modprobe habanalabs_en
sudo modprobe habanalabs_ib
sudo modprobe habanalabs_cn
sudo modprobe habanalabs
```
**Set up Container Usage**
To run containers, make sure to install and set up habanalabs-container-runtime as detailed in the below sections.

**Install Container Runtime**
The habanalabs-container-runtime is a modified runc that installs the container runtime library. This provides you the ability to select the devices to be mounted in the container. You only need to specify the indices of the devices for the container, and the container runtime will handle the rest. The habanalabs-container-runtime can support both Docker and Kubernetes.

**Package Retrieval:**

1. Download and install the public key:
```
curl -X GET https://vault.habana.ai/artifactory/api/gpg/key/public | sudo apt-key add --
```

2. Get the name of the operating system:

```
lsb_release -c | awk '{print $2}'
```

3. Create an apt source file /etc/apt/sources.list.d/artifactory.list with deb https://vault.habana.ai/artifactory/debian <OS name from previous step> main content.

In our case, it shows like this.
```
ubuntu@denvrbm-1099:~$ cat /etc/apt/sources.list.d/artifactory.list
deb https://vault.habana.ai/artifactory/debian jammy main
```

4. Update Debian cache:

```
sudo dpkg --configure -a

sudo apt-get update
```

5. Install habanalabs-container-runtime:
```
sudo apt install -y habanalabs-container-runtime
```
**Set up Container Runtime for Docker Engine**
1. Register habana runtime by adding the following to /etc/docker/daemon.json:
```
sudo tee /etc/docker/daemon.json <<EOF
{
   "runtimes": {
      "habana": {
            "path": "/usr/bin/habana-container-runtime",
            "runtimeArgs": []
      }
   }
}
EOF
```
2. Reconfigure the default runtime by adding the following to /etc/docker/daemon.json:
`"default-runtime": "habana"`
Your code should look similar to this:
```
{
   "default-runtime": "habana",
   "runtimes": {
      "habana": {
         "path": "/usr/bin/habana-container-runtime",
         "runtimeArgs": []
      }
   }
}
```
3. Restart Docker:
```
sudo systemctl restart docker
```

## Llama2 Inference on Gaudi2
In this step, we will launch the Habana Gaudi container. The we will log into the container and perform some software package installations and runnig the code for doing the inference. Before running the next set of instructions, you should be in the `home/ubuntu` folder.

Run the command `sudo su` to assume **root** privilege.

1. Pull docker container
```
docker pull vault.habana.ai/gaudi-docker/1.16.0/ubuntu22.04/habanalabs/pytorch-installer-2.2.2:latest
```

2. Launch the container and execute inside the container
```
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host vault.habana.ai/gaudi-docker/1.16.0/ubuntu22.04/habanalabs/pytorch-installer-2.2.2:latest
```

3. Install Ray, Optimum Habana and start the service. Version should be per the support matrix noted above.
```
pip install ray[tune,serve]==2.20.0
```
```
pip install git+https://github.com/huggingface/optimum-habana.git
```
**Replace 1.14.0 with the driver version of the container.**
This is needed if you are doing distributed inference with DeepSpeed
```
pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.14.0
```

**Only needed by the DeepSpeed example.**
```
export RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES=1
```

4. Start Ray
```
ray start --head
```

5. Export the Huggingface Token. Change `your_huggingface_token' with you actual token
```
export HUGGINGFACE_HUB_TOKEN="your_huggingface_token"
```

6. Caching the sharded model
Create a python program called `sharded_model.py`
```
vi sharded_model.py
```
Press `'i` for insert mode. Copy the lines of code below for the program `sharded_model.py` into the vi editor. On the keybord press these keys one by one - `Esc`,`:`,`w`,`q`,`!` then the `Enter` key. This will save the file and exit.

***sharded_model.py***
```
import os
from huggingface_hub import snapshot_download

# Get the Hugging Face token from the environment variable
token = os.getenv("HUGGINGFACE_HUB_TOKEN")

snapshot_download(
    "meta-llama/Llama-2-70b-chat-hf",
    cache_dir=os.getenv("TRANSFORMERS_CACHE", None),
    token=token
)
```
Now run the program. This is a large model. Might take a few mins to complete this step, please be patient.
```
python3 sharded_model.py
```

7. Create a python program called `intel_gaudi_inference_serve_deepspeed.py`
```
vi intel_gaudi_inference_serve_deepspeed.py
```
Press `'i` for insert mode. Copy the lines of code below for the program `intel_gaudi_inference_serve_deepspeed.py` into the vi editor. On the keybord press these keys one by one - `Esc`,`:`,`w`,`q`,`!` then the `Enter` key. This will save the file and exit.

***intel_gaudi_inference_serve_deepspeed.py***

```
import os
from huggingface_hub import snapshot_download
import tempfile
from typing import Dict, Any
from starlette.requests import Request
from starlette.responses import StreamingResponse

import torch
from transformers import TextStreamer

import ray
from ray import serve
from ray.util.queue import Queue
from ray.runtime_env import RuntimeEnv

# Set your Hugging Face token here or use an environment variable
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN", "your_huggingface_token_here")

# Ensure the model is downloaded with authentication
snapshot_download(
    "meta-llama/Llama-2-70b-chat-hf",
    cache_dir=os.getenv("TRANSFORMERS_CACHE", None),
    token=HUGGINGFACE_TOKEN
)

@ray.remote(resources={"HPU": 1})
class DeepSpeedInferenceWorker:
    def __init__(self, model_id_or_path: str, world_size: int, local_rank: int):
        """An actor that runs a DeepSpeed inference engine.

        Arguments:
            model_id_or_path: Either a Hugging Face model ID
                or a path to a cached model.
            world_size: Total number of worker processes.
            local_rank: Rank of this worker process.
                The rank 0 worker is the head worker.
        """
        from transformers import AutoTokenizer, AutoConfig
        from optimum.habana.transformers.modeling_utils import (
            adapt_transformers_to_gaudi,
        )

        # Tweak transformers for better performance on Gaudi.
        adapt_transformers_to_gaudi()

        self.model_id_or_path = model_id_or_path
        self._world_size = world_size
        self._local_rank = local_rank
        self.device = torch.device("hpu")

        self.model_config = AutoConfig.from_pretrained(
            model_id_or_path,
            torch_dtype=torch.bfloat16,
            token=HUGGINGFACE_TOKEN,
            trust_remote_code=False,
        )

        # Load and configure the tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id_or_path, use_fast=False, token=HUGGINGFACE_TOKEN
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        import habana_frameworks.torch.distributed.hccl as hccl

        # Initialize the distributed backend.
        hccl.initialize_distributed_hpu(
            world_size=world_size, rank=local_rank, local_rank=local_rank
        )
        torch.distributed.init_process_group(backend="hccl")

    def load_model(self):
        """Load the model to HPU and initialize the DeepSpeed inference engine."""

        import deepspeed
        from transformers import AutoModelForCausalLM
        from optimum.habana.checkpoint_utils import (
            get_ds_injection_policy,
            write_checkpoints_json,
        )

        # Construct the model with fake meta Tensors.
        # Loads the model weights from the checkpoint later.
        with deepspeed.OnDevice(dtype=torch.bfloat16, device="meta"):
            model = AutoModelForCausalLM.from_config(
                self.model_config, torch_dtype=torch.bfloat16
            )
        model = model.eval()

        # Create a file to indicate where the checkpoint is.
        checkpoints_json = tempfile.NamedTemporaryFile(suffix=".json", mode="w+")
        write_checkpoints_json(
            self.model_id_or_path, self._local_rank, checkpoints_json, token=HUGGINGFACE_TOKEN
        )

        # Prepare the DeepSpeed inference configuration.
        kwargs = {"dtype": torch.bfloat16}
        kwargs["checkpoint"] = checkpoints_json.name
        kwargs["tensor_parallel"] = {"tp_size": self._world_size}
        # Enable the HPU graph, similar to the cuda graph.
        kwargs["enable_cuda_graph"] = True
        # Specify the injection policy, required by DeepSpeed Tensor parallelism.
        kwargs["injection_policy"] = get_ds_injection_policy(self.model_config)

        # Initialize the inference engine.
        self.model = deepspeed.init_inference(model, **kwargs).module

    def tokenize(self, prompt: str):
        """Tokenize the input and move it to HPU."""

        input_tokens = self.tokenizer(prompt, return_tensors="pt", padding=True)
        return input_tokens.input_ids.to(device=self.device)

    def generate(self, prompt: str, **config: Dict[str, Any]):
        """Take in a prompt and generate a response."""

        input_ids = self.tokenize(prompt)
        gen_tokens = self.model.generate(input_ids, **config)
        return self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]

    def streaming_generate(self, prompt: str, streamer, **config: Dict[str, Any]):
        """Generate a streamed response given an input."""

        input_ids = self.tokenize(prompt)
        self.model.generate(input_ids, streamer=streamer, **config)

    def get_streamer(self):
        """Return a streamer.

        We only need the rank 0 worker's result.
        Other workers return a fake streamer.
        """

        if self._local_rank == 0:
            return RayTextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
        else:

            class FakeStreamer:
                def put(self, value):
                    pass

                def end(self):
                    pass

            return FakeStreamer()


class RayTextIteratorStreamer(TextStreamer):
    def __init__(
        self,
        tokenizer,
        skip_prompt: bool = False,
        timeout: int = None,
        **decode_kwargs: Dict[str, Any],
    ):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.text_queue = Queue()
        self.stop_signal = None
        self.timeout = timeout

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.text_queue.put(text, timeout=self.timeout)
        if stream_end:
            self.text_queue.put(self.stop_signal, timeout=self.timeout)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.text_queue.get(timeout=self.timeout)
        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value

# We need to set these variables for this example.
HABANA_ENVS = {
    "PT_HPU_LAZY_ACC_PAR_MODE": "0",
    "PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES": "0",
    "PT_HPU_ENABLE_WEIGHT_CPU_PERMUTE": "0",
    "PT_HPU_ENABLE_LAZY_COLLECTIVES": "true",
    "HABANA_VISIBLE_MODULES": "0,1,2,3,4,5,6,7",
}


# Define the Ray Serve deployment.
@serve.deployment
class DeepSpeedLlamaModel:
    def __init__(self, world_size: int, model_id_or_path: str):
        self._world_size = world_size

        # Create the DeepSpeed workers
        self.deepspeed_workers = []
        for i in range(world_size):
            self.deepspeed_workers.append(
                DeepSpeedInferenceWorker.options(
                    runtime_env=RuntimeEnv(env_vars=HABANA_ENVS)
                ).remote(model_id_or_path, world_size, i)
            )

        # Load the model to all workers.
        for worker in self.deepspeed_workers:
            worker.load_model.remote()

        # Get the workers' streamers.
        self.streamers = ray.get(
            [worker.get_streamer.remote() for worker in self.deepspeed_workers]
        )

    def generate(self, prompt: str, **config: Dict[str, Any]):
        """Send the prompt to workers for generation.

        Return after all workers finish the generation.
        Only return the rank 0 worker's result.
        """

        futures = [
            worker.generate.remote(prompt, **config)
            for worker in self.deepspeed_workers
        ]
        return ray.get(futures)[0]

    def streaming_generate(self, prompt: str, **config: Dict[str, Any]):
        """Send the prompt to workers for streaming generation.

        Only use the rank 0 worker's result.
        """

        for worker, streamer in zip(self.deepspeed_workers, self.streamers):
            worker.streaming_generate.remote(prompt, streamer, **config)

    def consume_streamer(self, streamer):
        """Consume the streamer and return a generator."""
        for token in streamer:
            yield token

    async def __call__(self, http_request: Request):
        """Handle received HTTP requests."""

        # Load fields from the request
        json_request: str = await http_request.json()
        text = json_request["text"]
        # Config used in generation
        config = json_request.get("config", {})
        streaming_response = json_request["stream"]

        # Prepare prompts
        prompts = []
        if isinstance(text, list):
            prompts.extend(text)
        else:
            prompts.append(text)

        # Process the configuration.
        config.setdefault("max_new_tokens", 128)

        # Enable HPU graph runtime.
        config["hpu_graphs"] = True
        # Lazy mode should be True when using HPU graphs.
        config["lazy_mode"] = True

        # Non-streaming case
        if not streaming_response:
            return self.generate(prompts, **config)

        # Streaming case
        self.streaming_generate(prompts, **config)
        return StreamingResponse(
            self.consume_streamer(self.streamers[0]),
            status_code=200,
            media_type="text/plain",
        )


# Replace the model ID with a path if necessary.
entrypoint = DeepSpeedLlamaModel.bind(8, "meta-llama/Llama-2-70b-chat-hf")
```

8. Start the deployment
```
serve run intel_gaudi_inference_serve_deepspeed:entrypoint
```

You should see `Deployed app 'default' successfully`. In our case, it showed something like this:
```
Loading 15 checkpoint shards: 100%|██████████| 15/15 [02:23<00:00,  9.02s/it] [repeated 8x across cluster]
(raylet) [2024-06-27 12:03:44,890 E 87977 88007] (raylet) file_system_monitor.cc:111: /tmp/ray/session_2024-06-27_12-00-40_170156_87593 is over 95% full, available space: 41949040640; capacity: 958463963136. Object creation will fail if spilling is required.
2024-06-27 12:03:47,406 INFO handle.py:126 -- Created DeploymentHandle 'oes2n3s8' for Deployment(name='DeepSpeedLlamaModel', app='default').
2024-06-27 12:03:47,407 INFO api.py:584 -- Deployed app 'default' successfully.
Loading 15 checkpoint shards: 100%|██████████| 15/15 [02:24<00:00,  9.64s/it] [repeated 13x across cluster]
(raylet) [2024-06-27 12:03:54,898 E 87977 88007] (raylet) file_system_monitor.cc:111: /tmp/ray/session_2024-06-27_12-00-40_170156_87593 is over 95% full, available space: 41949065216; capacity: 958463963136. Object creation will fail if spilling is required.
(raylet) [2024-06-27 12:04:04,906 E 87977 88007] (raylet) file_system_monitor.cc:111: /tmp/ray/session_2024-06-27_12-00-40_170156_87593 is over 95% full, available space: 41947672576; capacity: 958463963136. Object creation will fail if spilling is required.
```

**If you run the command `hl-smi`, you will see that the model has been loaded in all the 8 Gaudi2 accelerator cards memory within the node. Below is our output of running the `watch -n 1 hl-smi` command at this point.

```
Every 1.0s: hl-smi                                                                           denvrbm-1098: Thu Jun 27 12:06:24 2024

+-----------------------------------------------------------------------------+
| HL-SMI Version:                              hl-1.16.0-fw-50.1.2.0          |
| Driver Version:                                     1.16.2-f195ec4          |
|-------------------------------+----------------------+----------------------+
| AIP  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | AIP-Util  Compute M. |
|===============================+======================+======================|
|   0  HL-225              N/A  | 0000:19:00.0     N/A |                   0  |
| N/A   27C   N/A   102W / 600W |  17818MiB / 98304MiB |     0%           N/A |
|-------------------------------+----------------------+----------------------+
|   1  HL-225              N/A  | 0000:b3:00.0     N/A |                   0  |
| N/A   25C   N/A    92W / 600W |  17818MiB / 98304MiB |     0%           N/A |
|-------------------------------+----------------------+----------------------+
|   2  HL-225              N/A  | 0000:b4:00.0     N/A |                   0  |
| N/A   29C   N/A    99W / 600W |  17818MiB / 98304MiB |     0%           N/A |
|-------------------------------+----------------------+----------------------+
|   3  HL-225              N/A  | 0000:1a:00.0     N/A |                   0  |
| N/A   28C   N/A   103W / 600W |  17818MiB / 98304MiB |     0%           N/A |
|-------------------------------+----------------------+----------------------+
|   4  HL-225              N/A  | 0000:43:00.0     N/A |                   0  |
| N/A   27C   N/A    93W / 600W |  17818MiB / 98304MiB |     0%           N/A |
|-------------------------------+----------------------+----------------------+
|   5  HL-225              N/A  | 0000:44:00.0     N/A |                   0  |
| N/A   26C   N/A    99W / 600W |  17818MiB / 98304MiB |     0%           N/A |
|-------------------------------+----------------------+----------------------+
|   6  HL-225              N/A  | 0000:cc:00.0     N/A |                   0  |
| N/A   27C   N/A    96W / 600W |  17818MiB / 98304MiB |     0%           N/A |
|-------------------------------+----------------------+----------------------+
|   7  HL-225              N/A  | 0000:cd:00.0     N/A |                   0  |
| N/A   26C   N/A   104W / 600W |  17818MiB / 98304MiB |     0%           N/A |
|-------------------------------+----------------------+----------------------+
```

9. Open another shell terminal and log into the `home/ubuntu` folder. Then run the following commands.

Create a python program called `send_request.py`
```
vi send_request.py
```
Press `'i` for insert mode. Copy the lines of code below for the program `send_request.py` into the vi editor. On the keybord press these keys one by one - `Esc`,`:`,`w`,`q`,`!` then the `Enter` key. This will save the file and exit.

**send_request.py**
In the program below, you can change the prompt to anything you like.
```
import requests

# Prompt for the model
prompt = "When in Los Angeles, I should visit,"

# Add generation config here
config = {}

# Non-streaming response
sample_input = {"text": prompt, "config": config, "stream": False}
outputs = requests.post("http://127.0.0.1:8000/", json=sample_input, stream=False)
print(outputs.text, flush=True)

# Streaming response
sample_input["stream"] = True
outputs = requests.post("http://127.0.0.1:8000/", json=sample_input, stream=True)
outputs.raise_for_status()
for output in outputs.iter_content(chunk_size=None, decode_unicode=True):
    print(output, end="", flush=True)
print()
```
10. Run the inference request.
```
python3 send_request.py
```
You will be able to see the output from the Llama2 model. In our case the output came out like this.
```
ubuntu@denvrbm-1098:~$ python3 send_request.py
When in Los Angeles, I should visit, where?

Los Angeles is a city with a rich history, diverse culture, and endless entertainment options. From iconic landmarks to hidden gems, there's something for everyone in LA. Here are some must-visit places to add to your itinerary:

1. Hollywood: Walk along the Hollywood Walk of Fame, visit the TCL Chinese Theatre, and take a tour of movie studios like Paramount Pictures or Warner Bros.
2. Beverly Hills: Shop on Rodeo Drive, visit the Beverly Hills Hotel, and enjoy a meal at one
right?

Well, yes and no. While Los Angeles is a fantastic city with a lot to offer, it's not the only place worth visiting in California. Here are some alternative destinations to consider:

1. San Francisco: San Francisco is a charming and vibrant city with a rich history, cultural attractions, and a thriving food scene. Visit Fisherman's Wharf, Alcatraz Island, and the Golden Gate Bridge.
2. Yosemite National Park: Yosemite is a stunning national park located in the Sierra Nevada mountains
ubuntu@denvrbm-1098:~$
```

## Docker Container Cleanup
Run the commands below.

Stop running containers.
```
docker stop $(docker ps -aq)
```

Remove all containers
```
docker rm $(docker ps -aq)
```
## Acknowledgements
[Serve Llama2-7b/70b on a single or multiple Intel Gaudi Accelerator](https://docs.ray.io/en/latest/serve/tutorials/intel-gaudi-inference.html)
