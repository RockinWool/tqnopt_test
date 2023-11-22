import matplotlib.pyplot as plt
import pickle

# スクリプト1で保存したものを読み込み
with open("./data/figure3.pkl", "rb") as f:
    fig = pickle.load(f)

plt.show() # スクリプト1と同じものが表示されるはず