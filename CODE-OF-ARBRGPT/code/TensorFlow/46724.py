import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # To suppress TFLiteConverter warnings


class DummyMatmul(tf.Module):
    def __init__(self, matrix):
        super().__init__()
        self.matrix = tf.constant(matrix)

    @tf.function
    def __call__(self, signal_tensor):
        result = tf.matmul(signal_tensor, self.matrix)
        return -result


class DummyTensordot(tf.Module):
    def __init__(self, matrix):
        super().__init__()
        self.matrix = tf.constant(matrix)

    @tf.function
    def __call__(self, signal_tensor):
        result = tf.tensordot(signal_tensor, self.matrix, 1)
        return -result


def save_concrete_func(tf_module: tf.Module, input_spec: tf.TensorSpec, output_path):
    concrete_func = tf_module.__call__.get_concrete_function(signal_tensor=input_spec)
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.target_ops = [tf.lite.OpsSet.SELECT_TF_OPS, tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.experimental_enable_mlir_converter = True
    tflite_model = converter.convert()
    # Save the model.
    with open(output_path, 'wb') as f:
        f.write(tflite_model)


def run_interpreter(interpreter, input: np.ndarray):
    interpreter_out_indices = [o['index'] for o in interpreter.get_output_details()]
    interpreter.resize_tensor_input(0, input.shape)
    interpreter.allocate_tensors()
    interpreter.set_tensor(0, input)
    interpreter.invoke()
    interpreter_outs = [interpreter.get_tensor(interpreter_out_index) for interpreter_out_index in
                        interpreter_out_indices]
    return interpreter_outs


def dummy_test(matrix_shape=(128, 128)):
    np.random.seed(42)
    matrix = np.random.randint(0, 10, matrix_shape).astype(np.float32)
    np.random.seed(42)
    test_input = np.random.randint(0, 10, [1, 6, matrix_shape[0]]).astype(np.float32)
    print('Left matrix:')
    print(test_input)
    print('Right matrix:')
    print(matrix)

    signal_tensor_spec = tf.TensorSpec(shape=[1, None, matrix_shape[0]], dtype=tf.float32)

    dummy_matmul = DummyMatmul(matrix)
    dummy_tensordot = DummyTensordot(matrix)
    save_concrete_func(dummy_matmul, signal_tensor_spec, 'dummy_matmul.tflite')
    save_concrete_func(dummy_tensordot, signal_tensor_spec, 'dummy_tensordot.tflite')

    dummy_matmul_interpreter = tf.lite.Interpreter('dummy_matmul.tflite')
    dummy_tensordot_interpreter = tf.lite.Interpreter('dummy_tensordot.tflite')

    print("Invoking tf.Modules:")
    for i in range(3):
        matmul_out = [dummy_matmul(test_input)]
        tensordot_out = [dummy_tensordot(test_input)]
        # print(np.abs(matmul_out[0] - tensordot_out[0])) You can see the diffence with this line
        print(f'Iteration {i} - Maximum Absolute Difference:',
              [np.abs(mout - tout).max() for mout, tout in zip(matmul_out, tensordot_out)])
    print('- * ' * 20)


    print("Invoking interpreters:")
    for i in range(3):
        matmul_out = run_interpreter(dummy_matmul_interpreter, test_input)
        tensordot_out = run_interpreter(dummy_tensordot_interpreter, test_input)
        # print(np.abs(matmul_out[0] - tensordot_out[0])) You can see the diffence with this line
        print(f'Iteration {i} - Maximum Absolute Difference:',
              [np.abs(mout - tout).max() for mout, tout in zip(matmul_out, tensordot_out)])
    print('- * ' * 20)
