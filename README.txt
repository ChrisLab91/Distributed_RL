

vm setup

sudo apt-get install -y python3-numpy python3-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python3-opengl libboost-all-dev libsdl2-dev swig python3-git
sudo pip3 install gym[all]

sudo ufw allow 2222/tcp
sudo ufw allow ssh
sudo ufw enable

server specs

cluster=tf.train.ClusterSpec({
    "worker": [
        "10.155.208.112:2222"
    ],
    "ps": [
        "10.155.209.25:2222"
    ]})
server = tf.train.Server(cluster, job_name="worker", task_index=0)
server = tf.train.Server(cluster, job_name="ps", task_index=0)

