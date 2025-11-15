import os
from openai import OpenAI

# os.environ["DASHSCOPE_API_KEY"] = ""os.getenv("DASHSCOPE_API_KEY")

KEY = os.getenv("DASHSCOPE_API_KEY")

def get_chat_messages(prompt_text):
    client = OpenAI(
        api_key=KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    response = client.chat.completions.create(
        model="qwen3-235b-a22b",
        messages=prompt_text,
        extra_body={"enable_thinking": False},
        temperature=0.3,
        max_tokens=2000
    )
    return response.choices[0].message.content


def read_files_from_directory(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    return files


def process_files(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    files = read_files_from_directory(input_directory)
    for file_name in files:
        with open(os.path.join(input_directory, file_name), 'r', encoding='utf-8') as file:
            bug_report = file.read()
            bug_id = file_name.split('_')[0]

        # 定义对话信息
        action_prompt = "Available actions: [Install, Configure, Train, Monitor, Debug, Optimize, Save, Load, Test, Report]"
        object_prompt = "Available objects: [TensorFlow, Dataset, Model, Session, GPU Memory, Environment, Error Log, Configuration File, Checkpoint, TFRecord File]"
        roses_prompt = """
Step1-Step to reproduce, I want you to use keywords to describe the step to reproduce, and use the format of [Available actions] (Available objects).
Step2-requirements and reproduce code, output a self-contained example that can reproduce the issue, including a list of the environments needed to reproduce the code and the minimal reproducible test case.
Step3-Dockerfile, output the DockerFile that replicates the desired environment.  
Step4-running result, output the defect phenomenon in the format as follows.      
[symptom](Available Option: Build Failure, Crash, Hang, Incorrect Functionality, Poor Performance, Unreported)
[root_cause](Available Option: API Incompatibility, API Misuse, Concurrency, Dependent Module Issue, Environment Incompatibility, Incorrect Algorithm Implementation, Incorrect Assignment, Incorrect Exception Handling, Misconfiguration, Numerical Issue, Others, Tensor Shape Misalignment, Type Confusion)	
[component](Available Option: Environment, General Utility Implementation, Graph-Level Implementation, Operation Implementation, User-Level API)	
[stage](Available Option: deployment, install, preprocessing,training, utility)
You need to make sure that your output code is a complete and runnable file, please do not have syntax errors.
You should answer the question in exactly the following format(bug_id should be replace by a number):
{Step to reproduce}
{requirements.txt}
{bug_id.py}
{Dockerfile}
{running result}
            """

        # example_prompt = """Example Bug Report:
        # [Title] [Bug report] error when using weight_norm and DataParallel at the same time.
        #
        # ## Issue description
        #
        # When I try to use weight_norm (dim=None) and DataParallel to use multiple gpus at the same time, there is an error:
        # ![image](https://user-images.githubusercontent.com/10334851/40043728-f9492b84-5857-11e8-960f-a8e88e8b5913.png)
        #
        #
        #
        # After digging into the code,  I found the reason is that the "weight_g" in weight_norm (dim=None) is a 0-dim tensor. This is due to the line 10 in torch/nn/utils/weight_norm.py: `return p.norm()`.
        # `norm()` returns a 0-dim tensor (scalar) in pytorch0.4.0, while in pytorch0.3.0, it returns a 1-dim tensor.
        # The 0-dim "weight_g" somehow generates the above error when replicating across multiple gpus as in the line 12 of torch/nn/parallel/replicate.py: "param_copies = Broadcast.apply(devices, *params)"
        #
        # for now, my solution is to reshape the "weight_g" into a 1-dim tensor by changing `return p.norm()` in the line 10 of torch/nn/utils/weight_norm.py into `return p.norm().view(-1)`. It solves the error.
        #
        # ## Code example
        # ```
        # import os
        # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
        # import torch
        # from torch import nn
        # from torch.nn.utils import weight_norm
        #
        # device = torch.device('cuda')
        # model = weight_norm(nn.Linear(20, 30), dim=None)
        # model = nn.DataParallel(model).to(device)
        #
        # x = torch.rand(40, 20).to(device)
        # y = model(x)
        # loss = y.mean()
        # loss.backward()
        # ```
        #
        # ## System Info
        #
        # - PyTorch or Caffe2: PyTorch
        # - How you installed PyTorch (conda, pip, source): pip
        # - Build command you used (if compiling from source):
        # - OS: Ubuntu 16.04
        # - PyTorch version: 0.4.0
        # - Python version: 2.7
        # - CUDA/cuDNN version: 8.0
        # - GPU models and configuration:
        # - GCC version (if compiling from source):
        # - CMake version:
        # - Versions of any other relevant libraries:
        #
        # """
        #
        # cot_prompt = """Example Answer:
        # 1st step is "Import os, torch, torch.nn as nn, and torch.nn.utils.weight_norm".
        # The action is "import" and the target components are "os", "torch", "torch.nn as nn", and "torch.nn.utils.weight_norm".
        # Following the Action primitives, the entity of the step is:
        # [Import] ["os", "torch", "torch.nn as nn", "torch.nn.utils.weight_norm"]
        # 2nd step is "Set the environment variable CUDA_VISIBLE_DEVICES to '0,1'".
        # The action is "set" and the target component is "CUDA_VISIBLE_DEVICES='0,1'".
        # This controls which GPUs will be visible to the script when using CUDA.
        # Entity: [SetEnv] ["CUDA_VISIBLE_DEVICES='0,1'"]
        # 3rd step is "Create a model with weight normalization applied (dim=None)".
        # The action is "create" and the target component is "weight_norm(nn.Linear(20, 30), dim=None)".
        # This constructs a linear layer and applies weight normalization without specifying a dimension.
        # Entity: [CreateModel] ["weight_norm(nn.Linear(20, 30), dim=None)"]
        # 4th step is "Parallelize the model across multiple GPUs using DataParallel".
        # The action is "parallelize" and the target component is "nn.DataParallel(model).to(device)".
        # This wraps the model to allow multi-GPU computation.
        # Entity: [ParallelizeModel] ["nn.DataParallel(model).to(device)"]
        # 5th step is "Prepare input tensor on the GPU device".
        # The action is "define" and the target component is "torch.rand(40, 20).to(device)".
        # This creates a random input tensor and moves it to the selected device (GPU).
        # Entity: [DefineInput] ["torch.rand(40, 20).to(device)"]
        # 6th step is "Perform forward pass through the model".
        # The action is "forward" and the target component is "model(x)".
        # This computes the output of the model given the input x.
        # Entity: [ForwardPass] ["model(x)"]
        # 7th step is "Compute the loss by taking the mean of the output".
        # The action is "calculate" and the target component is "y.mean()".
        # This scalar value represents the loss used for backpropagation.
        # Entity: [CalculateLoss] ["y.mean()"]
        # 8th step is "Perform backward pass to compute gradients".
        # The action is "backward" and the target component is "loss.backward()".
        # This computes the gradients for all parameters involved in the computation graph.
        # Entity: [BackwardPass] ["loss.backward()"]
        #
        # Overall, the extracted S2R entities are:
        # {Step to reproduce}
        # 1. [Import] ("os", "torch", "torch.nn as nn", "torch.nn.utils.weight_norm")
        # 2. [SetEnv] ("CUDA_VISIBLE_DEVICES='0,1'")
        # 3. [CreateModel] ("weight_norm(nn.Linear(20, 30), dim=None)")
        # 4. [ParallelizeModel] ("nn.DataParallel(model).to(device)")
        # 5. [PrepareInput] ("torch.rand(40, 20).to(device)")
        # 6. [ForwardPass] ("model(x)")
        # 7. [CalculateLoss] ("y.mean()")
        # 8. [BackwardPass] ("loss.backward()")
        # """
        # one_shot_answer = cot_prompt + "\n" + """
        # {requirements.txt}
        # torch==0.4.0
        #
        # {7568.py}
        # import os
        # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
        # import torch
        # from torch import nn
        # from torch.nn.utils import weight_norm
        #
        # device = torch.device('cuda')
        # model = weight_norm(nn.Linear(20, 30), dim=None)
        # model = nn.DataParallel(model).to(device)
        #
        # x = torch.rand(40, 20).to(device)
        # y = model(x)
        # loss = y.mean()
        # loss.backward()
        #
        # {Dockerfile}
        # FROM nvidia/cuda:8.0-cudnn6-runtime-ubuntu16.04
        #
        # RUN apt-get update && \
        # apt-get install -y --no-install-recommends \
        # python2.7 \
        # python-pip \
        # && rm -rf /var/lib/apt/lists/*
        #
        # RUN ln -s /usr/bin/python2.7 /usr/bin/python
        #
        # COPY requirements.txt .
        # RUN pip install --upgrade pip && \
        # pip install -r requirements.txt --no-cache-dir
        #
        # COPY bug_id.py /app/bug_id.py
        #
        # CMD ["python", "/app/bug_id.py"]
        #
        # {running result}
        # [symptom](Crash)
        # [root_cause](Tensor Shape Misalignment)
        # [component](Graph-Level Implementation)
        # [stage](training)
        # """

        example_prompt = """Example Bug Report: 
**System information**

- OS Platform and Distribution (e.g., Linux Ubuntu 16.04): Debian Testing

- TensorFlow installed from (source or binary): Binary

- TensorFlow version (use command below): 2.0.0.dev20190227

- Python version: 3.7





**Describe the current behavior**



A function which correctly works when in eager mode does not work anymore when annotated with `tf.function`.



In particular, it complains about `ValueError: tf.function-decorated function tried to create variables on non-first call.`, even though the function is always called with different parameters.



This is a continuation of https://github.com/tensorflow/tensorflow/issues/26812#issuecomment-475600836.



**Describe the expected behavior**



The `apply_gradients_once()` function should work even when annotated with `tf.function`.



**Code to reproduce the issue**



```python3

import tensorflow as tf

import numpy as np





fast_optimizer = tf.keras.optimizers.Adam(

        learning_rate=1e-3)



slow_optimizer = tf.keras.optimizers.Adam(

        learning_rate=1e-3 * 1e-9)





@tf.function

def apply_gradients_once(optimizer, grads, vars):

    grads = [grads]

    optimizer.apply_gradients(zip(grads, vars))





def apply_grads(use_fast, grads_per_model, vars):

    for i in range(2):

        if use_fast[i]:

            apply_gradients_once(fast_optimizer, grads_per_model[i], vars[i])

        else:

            apply_gradients_once(slow_optimizer, grads_per_model[i], vars[i])





def compute_loss(w, x, y):

    r = (w * x - y)**2

    r = tf.math.reduce_mean(r)

    return r



def compute_gradients(model):

    with tf.GradientTape() as tape:

        tape.watch(model)

        loss = compute_loss(model, x, y)

    grads = tape.gradient(loss, model)

    return grads





w = [

    tf.Variable(0.0),

    tf.Variable(1.0)]



x = np.array([1, 2, 3])

y = np.array([1, 2, 3])



vars = []

grads = []

for i in range(2):

    vars.append([w[i]])

    grads.append(compute_gradients(w[i]))



apply_grads([True, False], grads, vars)

```





**Other info / logs**



Error log:



```

Traceback (most recent call last):

  File "main.py", line 52, in <module>

    apply_grads([True, False], grads, vars)

  File "main.py", line 23, in apply_grads

    apply_gradients_once(slow_optimizer, grads_per_model[i], vars[i])

  File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 414, in __call__

    return self._stateless_fn(*args, **kwds)  # pylint: disable=not-callable

  File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 1254, in __call__

    graph_function, args, kwargs = self._maybe_define_function(args, kwargs)

  File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 1577, in _maybe_define_function

    args, kwargs, override_flat_arg_shapes=relaxed_arg_shapes)

  File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/function.py", line 1479, in _create_graph_function

    capture_by_value=self._capture_by_value),

  File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/func_graph.py", line 685, in func_graph_from_py_func

    func_outputs = python_func(*func_args, **func_kwargs)

  File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 317, in wrapped_fn

    return weak_wrapped_fn().__wrapped__(*args, **kwds)

  File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/func_graph.py", line 677, in wrapper

    ), args, kwargs)

  File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/autograph/impl/api.py", line 392, in converted_call

    result = converted_f(*effective_args, **kwargs)

  File "/tmp/tmpr2ti5o1e.py", line 4, in tf__apply_gradients_once

    ag__.converted_call('apply_gradients', optimizer, ag__.ConversionOptions(recursive=True, verbose=0, strip_decorators=(tf.function, defun, ag__.convert, ag__.do_not_convert, ag__.converted_call), force_conversion=False, optional_features=(), internal_convert_user_code=True), (zip(grads, vars),), {})

  File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/autograph/impl/api.py", line 267, in converted_call

    return _call_unconverted(f, args, kwargs)

  File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/autograph/impl/api.py", line 188, in _call_unconverted

    return f(*args, **kwargs)

  File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/keras/optimizer_v2/optimizer_v2.py", line 399, in apply_gradients

    self._create_hypers()

  File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/keras/optimizer_v2/optimizer_v2.py", line 558, in _create_hypers

    aggregation=tf_variables.VariableAggregation.ONLY_FIRST_REPLICA)

  File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/keras/optimizer_v2/optimizer_v2.py", line 727, in add_weight

    aggregation=aggregation)

  File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/training/tracking/base.py", line 622, in _add_variable_with_custom_getter

    **kwargs_for_getter)

  File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/keras/engine/base_layer_utils.py", line 152, in make_variable

    aggregation=aggregation)

  File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/ops/variables.py", line 212, in __call__

    return cls._variable_v1_call(*args, **kwargs)

  File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/ops/variables.py", line 175, in _variable_v1_call

    aggregation=aggregation)

  File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/ops/variables.py", line 58, in getter

    return captured_getter(captured_previous, **kwargs)

  File "/usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/def_function.py", line 375, in invalid_creator_scope

    "tf.function-decorated function tried to create "

ValueError: tf.function-decorated function tried to create variables on non-first call.

    ```
"""

        one_shot_answer = """
{Step to reproduce}
1. [Import] ("tensorflow", "numpy")
2. [Initialize] ("fast_optimizer", "slow_optimizer")
3. [DefineFunction] ("apply_gradients_once") 
4. [CreateFunction] ("apply_grads")
5. [DefineFunction] ("compute_loss")
6. [DefineFunction] ("compute_gradients")
7. [Initialize] ("w[0]", "w[1]")
8. [PrepareInput] ("x", "y")
9. [ComputeGradients] ("compute_gradients")
10. [ApplyGradients] ("apply_grads")

{requirements.txt}
tensorflow==2.0.0
protobuf==3.20.3

{27120.py}
import tensorflow as tf
import numpy as np

fast_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
slow_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3 * 1e-9)

@tf.function

def apply_gradients_once(optimizer, grads, vars):
    grads = [grads]
    optimizer.apply_gradients(zip(grads, vars))

def apply_grads(use_fast, grads_per_model, vars):

    for i in range(2):
        if use_fast[i]:
            apply_gradients_once(fast_optimizer, grads_per_model[i], vars[i])
        else:
            apply_gradients_once(slow_optimizer, grads_per_model[i], vars[i])

def compute_loss(w, x, y):

    r = (w * x - y)**2
    r = tf.math.reduce_mean(r)
    return r

def compute_gradients(model):        
    with tf.GradientTape() as tape:        
        tape.watch(model)        
        loss = compute_loss(model, x, y)        
    grads = tape.gradient(loss, model)

    return grads

w = [tf.Variable(0.0), tf.Variable(1.0)]        
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])

vars = []  
grads = []
for i in range(2): 
    vars.append([w[i]])
    grads.append(compute_gradients(w[i]))

apply_grads([True, False], grads, vars)

{Dockerfile}
# 使用官方的Python基础镜像
FROM python:3.7-slim

# 设置工作目录
WORKDIR /usr/src/app

# 更新包列表并安装必要的依赖项
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# 安装Python依赖
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 复制当前目录下的所有文件到工作目录
COPY . .

# 指定启动容器时运行的命令
CMD ["python", "./27120.py"]



{running result}
[symptom](Crash)
[root_cause](Others)	
[component](User-Level API)	
[stage](training)
            """

        question_prompt = """Question: 
Bug Report: {}
Bug ID: {}
            """.format(bug_report, bug_id)

        prompt_text = [{"role": "system",
                        "content": "You are a senior bug report expert.Please remember your duty is to understand the bug report and identifying the steps to reproduce the issue."},
                       {"role": "user",
                        "content": action_prompt + "\n" + object_prompt + "\n" + roses_prompt},
                       {"role": "assistant",
                        "content": "I see. I will generate the answer in formatted string -> {Step to reproduce}\n{requirements.txt}\n{bug_id.py}\n{running result}"},
                       {"role": "user",
                        "content": example_prompt},
                       {"role": "assistant",
                        "content": one_shot_answer},
                       {"role": "user",
                        "content": question_prompt}
                       ]
        response_content = get_chat_messages(prompt_text)
        # 打印生成的回复
        print("ChatGPT：", response_content)
        output_file_name = os.path.splitext(file_name)[0] + '_response.txt'
        with open(os.path.join(output_directory, output_file_name), 'w', encoding='utf-8') as output_file:
            output_file.write(response_content)


# 使用示例
for framework in ['PyTorch', 'TensorFlow', 'MXNet']:
    input_directory = f'./input/{framework}_bugreports/summary_with_code'
    output_directory = (f'./output/{framework}/qwen_output_OS_KW_S2R')
    process_files(input_directory, output_directory)
# framework = 'TensorFlow'
# input_directory = f'./input/{framework}_bugreports/summary_with_code'
# output_directory = (f'./output/{framework}/output_OS_KW_S2R_CoT1')
# process_files(input_directory, output_directory)


