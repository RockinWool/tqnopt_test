import tensorflow as tf
# from tf_agents.agents.dqn import dqn_agent
# from tf_agents.environments import suite_gym
# from tensorflow.keras.utils import plot_model
import numpy as np

class QNetwork:
    def __init__(self,learning_rate,load_mode = False):
        self.model = tf.keras.models.load_model("./data/test_model_50000.h5")
        
  
