import tensorflow as tf

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# cuda version: release 10.1, V10.1.243

#$ cat /home/sruiz/local/bin/cuda/version.txt
#CUDA Version 10.1.243


# cuda driver version: 450.66
# tensorflow version 1.13

# nvidia-smi: NVIDIA-SMI 450.66       Driver Version: 450.66       CUDA Version: 11.0 