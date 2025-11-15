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
        roses_prompt = """
Step1-Step to reproduce, I want you to describe the step to reproduce. You must explain your thinking and reasoning process step by step. 
Step2-requirements and reproduce code, output a self-contained example that can reproduce the issue, including a list of the environments needed to reproduce the code and the minimal reproducible test case.
Step3-Dockerfile, output the DockerFile that replicates the desired environment.  
Step4-running result, output the defect phenomenon.      
You need to make sure that your output code is a complete and runnable file, please do not have syntax errors.
You should answer the question in exactly the following format(bug_id should be replace by a number):
{Step to reproduce}
{requirements.txt}
{bug_id.py}
{Dockerfile}
{running result}
            """


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

        cot_prompt = """Example Answer: 
1st step is "Import necessary libraries".
Reasoning: To begin any TensorFlow operation or model training, relevant libraries need to be imported.

2nd step is "Initialize optimizers for fast and slow learning rates".
Reasoning: Different parts of the model may require different learning rates. Here, two Adam optimizers with distinct learning rates are initialized.

3rd step is "Define the apply_gradients_once function decorated with @tf.function". 
Reasoning: This function applies gradients to variables using an optimizer. It is decorated with @tf.function for performance optimization through graph execution.

4th step is "Create a function apply_grads to handle gradient application based on condition".
Reasoning: This function decides which optimizer to use for applying gradients based on the use_fast parameter.

5th step is "Define the compute_loss function".
Reasoning: A loss function is needed to measure how well the model's predictions match the actual targets. This function computes the mean squared error between predictions and true values.

6th step is "Define the compute_gradients function using GradientTape".
Reasoning: In TensorFlow, GradientTape is used to record operations for automatic differentiation. This function calculates the gradients of the loss with respect to the model parameters.

7th step is "Initialize weights as tf.Variable".
Reasoning: Model weights must be initialized before training. Here, two weights are initialized as TensorFlow Variables.

8th step is "Prepare input data".
Reasoning: Input data needs to be defined for feeding into the model during training.

9th step is "Compute gradients for each model".
Reasoning: Gradients are computed for each model to update the weights in the direction that minimizes the loss.

10th step is "Apply gradients to models based on conditions".
Reasoning: Finally, gradients are applied to the model parameters according to whether a fast or slow optimizer should be used.

Overall, the extracted S2R entities are:
{Step to reproduce}
1. Import necessary libraries for TensorFlow operations and numerical computations.
2. Initialize two Adam optimizers for fast and slow learning rates respectively.
3. Define a function `apply_gradients_once(optimizer, grads, vars)` to apply gradients to variables, decorated with `@tf.function` for performance optimization.
4. Create a function `apply_grads(use_fast, grads_per_model, vars)` that applies gradients using either the fast or slow optimizer based on conditions.
5. Define a loss computation function `compute_loss(w, x, y)` to calculate the mean squared error between model predictions and actual values.
6. Define a function `compute_gradients(model)` using `GradientTape` to record operations for automatic differentiation and compute gradients.
7. Initialize two weights as `tf.Variable` in preparation for training.
8. Prepare input data `x` and target data `y`.
9. Compute gradients for each model by calling `compute_gradients(w[i])`.
10. Apply gradients to model parameters based on conditions by calling `apply_grads([True, False], grads, vars)`.
            """
        one_shot_answer = cot_prompt + "\n" + """
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
- [symptom]: The script fails with a ValueError when trying to apply gradients using `apply_gradients_once()` function decorated with `@tf.function`.
- [root_cause]: The error arises because the `@tf.function`-decorated function attempts to create variables on non-first call, which is not allowed. This issue occurs even when the function is called with different parameters.
- [component]: The problematic component involves the interaction between TensorFlow's `tf.function`, gradient application through optimizers (`Adam`), and variable creation within these functions.
- [stage]: The failure happens during the training stage when `apply_grads([True, False], grads, vars)` is invoked, specifically at the point where `optimizer.apply_gradients(zip(grads, vars))` is called inside the `apply_gradients_once` function.

            """

        question_prompt = """Question: 
Bug Report: {}
Bug ID: {}
            """.format(bug_report, bug_id)

        prompt_text = [{"role": "system",
                        "content": "You are a senior bug report expert.Please remember your duty is to understand the bug report and identifying the steps to reproduce the issue."},
                       {"role": "user",
                        "content": roses_prompt},
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
    output_directory = (f'./output/{framework}/qwen_output_OS_S2R_CoT')
    process_files(input_directory, output_directory)
# framework = 'TensorFlow'
# input_directory = f'./input/{framework}_bugreports/summary_with_code'
# output_directory = (f'./output/{framework}/output_OS_KW_S2R_CoT1')
# process_files(input_directory, output_directory)


