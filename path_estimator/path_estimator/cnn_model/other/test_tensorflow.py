import tensorflow as tf

if __name__=="__main__":
    tensor1 = tf.constant([0.0, 0.0, 1.0, 1.0])
    tensor2 = tf.constant([-0.5, -0.5, 0.5, 0.5])
    tensor3 = tf.constant([[1.0, 0.0],
                           [2.0, 0.0]])
    abs_tensor = tf.abs(tensor1 - tensor2)
    print(abs_tensor)
    print(tf.reduce_mean(abs_tensor))
    print(tf.reduce_mean(tensor3))