import argparse
import sys

import tensorflow as tf
import time

###Create local cluster for given number of worker and starts worker server with given index

FLAGS = None

def main(_):
  print(FLAGS.workers)
  ps_hosts=[]
  for i in range (2222, 2222+FLAGS.workers):
      ps_hosts.append('localhost:%d' % i)
  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"local": ps_hosts})

  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           'local',
                           task_index=FLAGS.task_index)
  server.join()
#  while True:
#      time.sleep(1)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  # Flags for defining the tf.train.ClusterSpec
  parser.add_argument(
      "--servers",
      type=int,
      default="1",
      help="total Number of Servers"
  )
  # Flags for defining the tf.train.Server
  parser.add_argument(
      "--task_index",
      type=int,
      default=0,
      help="Index of task within the job"
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)