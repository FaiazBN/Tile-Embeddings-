import tensorflow as tf
#from tensorflow.python.platform import build_info as tf_build_info

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("GPU Devices:", tf.config.list_physical_devices('GPU'))

# Try a small GPU computation
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
    print("Result of GPU computation:", tf.matmul(a, b))



print("TensorFlow version:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("GPU Available (legacy):", tf.test.is_gpu_available(cuda_only=True))

print("\nDevices:")
print(tf.config.list_physical_devices())

print("\nVisible GPUs:")
print(tf.config.list_physical_devices('GPU'))

print("\nGPU Details:")
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    print(gpu)