import tensorflow as tf





def main():

    graph = tf.Graph()



    # Create a basic graph with only an overlap_and_add on some random data.

    shape = (1, 1, 1024, 512)

    with graph.as_default():

        _input = tf.random.uniform(shape)

        ola = tf.signal.overlap_and_add(_input, 256, name='output')



    # Try executing that graph in regular TensorFlow.

    with tf.compat.v1.Session(graph=graph) as session:

        print(f"With regular TensorFlow, result is: {session.run(ola)}")



    # Convert to TFLite (using the V1 interface for simplicity)

    converter = tf.compat.v1.lite.TFLiteConverter(graph.as_graph_def(), [_input], [ola])

    tflite_model = converter.convert()



    # Write the model to disk...

    model_path = f'./{__file__}.tflite'

    with open(model_path, 'wb') as f:

        f.write(tflite_model)



    # ...so that we can load it into an interpreter and see the error!

    interpreter = tf.lite.Interpreter(model_path=model_path)

    interpreter.allocate_tensors()



    # This line should throw (as of tf-nightly-cpu-2.4.0.dev20200818):

    #   RuntimeError: tensorflow/lite/kernels/reshape.cc:66 \

    #   num_input_elements != num_output_elements (0 != 524800) \

    #   Node number 5 (RESHAPE) failed to prepare.

    interpreter.invoke()

    print("If we got here, the bug did not appear!")





if __name__ == "__main__":

    main()