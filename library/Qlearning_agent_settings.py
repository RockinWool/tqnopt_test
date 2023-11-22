import numpy as np
# 状態に応じて、行動を決定するクラス
class Actor:
    def get_action(self, state, episode, mainQN):   # [C]ｔ＋１での行動を返す
        # 徐々に最適行動のみをとる、ε-greedy法
        epsilon = 0.001 + 0.9 / (1.0+episode)
        if epsilon <= np.random.uniform(0, 1):
            return_TargetQs = mainQN.model.predict(np.array([state for i in range(2)]),verbose=0)[0]
            # action = float(np.argmax(return_TargetQs))  # 最大の報酬を返す行動を選択する
            action = return_TargetQs[0]
        else:
            #action = np.random.choice([0, 1])  # ランダムに行動する
            action = np.random.rand()
        return action