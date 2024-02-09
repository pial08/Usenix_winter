

# LLM-Powered Code Vulnerability Repair with Reinforcement Learning and Semantic Reward 

In software development, the predominant emphasis on functionality often supersedes security concerns, a trend gaining momentum with AI-driven automation tools like GitHub Copilot. These tools significantly improve developers' efficiency in functional code development. Nevertheless, it remains a notable concern that such tools are also responsible for creating insecure code, predominantly because of pre-training on publicly available repositories with vulnerable code. Moreover, developers are called the "weakest link in the chain" since they have very minimal knowledge of code security. Although existing solutions provide a reasonable solution to vulnerable code, they must adequately describe and educate the developers on code security to ensure that the security issues are not repeated. 
Therefore we introduce a multipurpose code vulnerability analysis system \texttt{SecRepair}, powered by a large language model, CodeGen2 assisting the developer in identifying and generating fixed code along with a complete description of the vulnerability with a code comment. Our innovative methodology uses a reinforcement learning paradigm to generate code comments augmented by a semantic reward mechanism. Inspired by how humans fix code issues, we propose an instruction-based dataset suitable for vulnerability analysis with LLMs. We further identify zero-day and N-day vulnerabilities in 6 Open Source  IoT Operating Systems on GitHub. Our findings underscore that incorporating reinforcement learning coupled with semantic reward augments our model's performance, thereby fortifying its capacity to address code vulnerabilities with improved efficacy.





#### Requirements
- Python 	3.7
- Pytorch 	1.9 
- Transformer 	4.4
- torchmetrics 0.11.4
- tree-sitter 0.20.1
- sctokenizer 0.0.8

Moreover the above libraries can be installed by the commands from *requirements.txt* file. It is assumed that the installation will be done in a Linux system with a GPU. If GPU does not exist please remove the first command from the *requirements.txt*  file and replace it with 

`conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 -c pytorch` for OSX

or 


`conda install pytorch==1.9.0 torchvision==0.10.1 torchaudio==0.9.1 cpuonly -c pytorch` for Linux and Windows with no GPU.

Instructions to install libraries using *requirements.txt* file.

```shell
cd code 
pip install -r requirements.txt
```


### Usage
The repository is partially based on [CodeXGLUE](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection).


### Training and Evaluation
The following command should be used for training, testing and evaluation. Please set the ```--output_dir``` to the address where the model will be saved. We have also compiled a shell file with the same command for ease of use for the practitioners. Please put the location/address of train, evaluation and test file directory for the parameters
```--train_data_file```, ```--eval_data_file``` and ```--test_data_file```. 


Please run the following commands:

```shell


./run.sh

or,

torchrun --nproc_per_node=4 --master_port=1234 train.py \
    --model_name_or_path Salesforce/codegen2-7B \
    --data_path <data_path> \
    --output_dir <model_name> \
    --num_train_epochs 4 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --gradient_checkpointing false \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --fsdp "full_shard offload auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'CodeGenBlock' \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fp16 True

```




### Datasets
- Please look at "cwe_dataset.csv" file for the dataset.


- Our N-day and zero-day samples are also available in the previous link under *Testing* directory.
- After downloading VulF dataset, please put it under the directory *data*.

### Inference
For inference, please run the evaluation_repair.py file. Inside the file, please change the model_name and dataset location accordingly.

## Cite  
Please cite the paper whenever our ReGVD is used to produce published results or incorporated into other software:

 



		

## License
As a free open-source implementation, our repository is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. All other warranties including, but not limited to, merchantability and fitness for purpose, whether express, implied, or arising by operation of law, course of dealing, or trade usage are hereby disclaimed. I believe that the programs compute what I claim they compute, but I do not guarantee this. The programs may be poorly and inconsistently documented and may contain undocumented components, features or modifications. I make no guarantee that these programs will be suitable for any application.

