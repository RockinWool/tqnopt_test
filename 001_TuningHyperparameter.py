import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  
import optuna
import tensorflow as tf
import numpy as np
from library.Data_preparing import Data_preparing as dp


class TQNetwork:
    def __init__(self,load_mode = False):
        # Initialize model
        self.model = []
        self.best_mse = 100
        if load_mode:
            self.load_model()
        else:
            self.run_study_main()

    def load_model(self):
        self.model = tf.keras.models.load_model("./data/best_model_cpu.h5")
    
    def create_model(self,trial):
        # We optimize the numbers of layers, their units and weight decay parameter.
        n_layers = trial.suggest_int("n_layers", 1, 3)
        weight_decay = trial.suggest_float("weight_decay", 1e-10, 1e-3, log=True)
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten())
        for i in range(n_layers):
            num_hidden = trial.suggest_int("n_units_l{}".format(i), 4, 128, log=True)
            model.add(
                tf.keras.layers.Dense(
                    num_hidden,
                    activation="relu",
                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                )
            )
        model.add(
            tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
        )
        return model

    def create_optimizer(self,trial):
        # We optimize the choice of optimizers as well as their parameters.
        kwargs = {}
        optimizer_options = ["RMSprop", "Adam", "SGD"]
        optimizer_selected = trial.suggest_categorical("optimizer", optimizer_options)
        if optimizer_selected == "RMSprop":
            kwargs["learning_rate"] = trial.suggest_float(
                "rmsprop_learning_rate", 1e-5, 1e-1, log=True
            )
            kwargs["weight_decay"] = trial.suggest_float("rmsprop_weight_decay", 0.85, 0.99)
            kwargs["momentum"] = trial.suggest_float("rmsprop_momentum", 1e-5, 1e-1, log=True)
        elif optimizer_selected == "Adam":
            kwargs["learning_rate"] = trial.suggest_float("adam_learning_rate", 1e-5, 1e-1, log=True)
        elif optimizer_selected == "SGD":
            kwargs["learning_rate"] = trial.suggest_float(
                "sgd_opt_learning_rate", 1e-5, 1e-1, log=True
            )
            kwargs["momentum"] = trial.suggest_float("sgd_opt_momentum", 1e-5, 1e-1, log=True)

        optimizer = getattr(tf.optimizers, optimizer_selected)(**kwargs)
        return optimizer

    def learn(self, model, optimizer, dataset, mode="eval"):
        mse = tf.metrics.MeanSquaredError(name = "mse",dtype=tf.float64)

        for batch, (inputs, outputs) in enumerate(dataset):
            with tf.GradientTape() as tape:
                inputs = np.reshape(inputs,[1,2])
                outputs =tf.cast(outputs, tf.float32)
                predictions = model(inputs, training=(mode == "train"))

                loss_value = tf.reduce_mean(tf.square(predictions - outputs))
                if mode == "eval":
                    mse(outputs, predictions)
                else:
                    grads = tape.gradient(loss_value, model.variables)
                    optimizer.apply_gradients(zip(grads, model.variables))

        if mode == "eval":
            return mse
    
    def objective(self,trial):
        EPOCHS = 10
        N = 500
        test_N = 50

        inputs, outputs = dp.data_creator(N)
        test_inputs,test_outputs = dp.data_creator(test_N)

        dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs))
        test_dataset = tf.data.Dataset.from_tensor_slices((test_inputs, test_outputs))

        # Build model and optimizer.
        model = self.create_model(trial)
        optimizer = self.create_optimizer(trial)

        # Training and validating cycle.
        #with tf.device("/cpu:0"):
        with tf.device("/GPU:0"):
            for _ in range(EPOCHS):
                self.learn(model, optimizer, dataset, "train")

            mse = self.learn(model, optimizer, test_dataset,"eval")


        if mse.result() < self.best_mse:
            self.best_mse = mse.result()

        # Return last validation accuracy.
        return mse.result()


    def run_study_main(self):
        study = optuna.create_study(direction="minimize",storage="sqlite:///data/done_study_cpu.db")
        study.optimize(self.objective, n_trials=3)

        print("Number of finished trials: ", len(study.trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

if __name__ =="__main__":  
    test_class = TQNetwork()
    test_class.run_study_main()