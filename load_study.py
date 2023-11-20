import optuna
import tensorflow as tf
from library.Data_preparing import Data_preparing as dp



class load_and_learn:
    def __init__(self):
        #please check storage_name first. It is described when you run Tuning_Hyperparameter.py.
        self.loaded_study = optuna.load_study("no-name-eaaabe5c-c5a2-4c94-ae97-527779a861df", storage="sqlite:///data/done_study_gpu.db")
        print(len(self.loaded_study.trials))
        print(type(self.loaded_study))
        print(self.loaded_study.best_params)
        print(self.loaded_study.best_params["n_layers"])
    
    def create_model(self):
        # We optimize the numbers of layers, their units and weight decay parameter.
        n_layers = self.loaded_study.best_params["n_layers"]
        weight_decay = self.loaded_study.best_params["weight_decay"]

        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Flatten())
        
        for i in range(n_layers):
            temp_str = "n_units_l{}".format(i)
            num_hidden = self.loaded_study.best_params[temp_str]        
            self.model.add(
                tf.keras.layers.Dense(
                    num_hidden,
                    activation="relu",
                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                )
            )
        self.model.add(
            tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
        )
        optimizer = self.loaded_study.best_params["optimizer"]
        loss_fn = tf.keras.losses.MeanSquaredError()
        self.model.compile(loss=loss_fn, optimizer=optimizer)
        
           
    def learn_main(self):
        N = 500
        inputs, outputs = dp.data_creator(N)
        self.model.fit(inputs,outputs,epochs = 10)
        self.model.save("./data/test_model.h5")
        
    
if __name__ == "__main__":
    cl = load_and_learn()
    cl.create_model()
    cl.learn_main()