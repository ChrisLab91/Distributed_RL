with tf.Session() as sess:
    with tf.device('/cpu:0'):
        matrix1 = tf.constant([[3., 3.]])
        matrix2 = tf.constant([[2.], [2.]])
        product = tf.matmul(matrix1, matrix2)
        result = sess.run(product)
        print(result)

tensorboard --logdir=~/DILAB_git/Distributed_VM_example_tensorflowlogs
ssh -L 16006:127.0.0.1:6006 ubuntu@10.155.209.48
echo 'yes' | scp - r ~/Schreibtisch/Uni/Distributed_VM_example / guest@kspqur6utc6vnwog.myfritz.net:~
echo 'yes' | scp -r ubuntu@10.155.209.48:~/Distributed_VM_example_modelcheckpoints /home/adrian/Schreibtisch/
/home/adrian/Schreibtisch/Uni/Distributed_VM_example

export CUDA_VISIBLE_DEVICES=""