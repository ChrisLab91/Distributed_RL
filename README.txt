

vm setup

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

