import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

import tensorflow as tf
from library.Qlearning_agent_settings import Actor
from library.Qnetwork import QNetwork
from library.Calculate_payoff import Payoff_dealer as pad


class payoff_matrix_concave_surface:
    def __init__(self):
        print("描画方法を選択してください")
        print("1: AIベース, 2: 数式ベース, 3: 数式ベース(総利得、相手利得込み)")
        self.drawtype = input(">>")
        #create figure
        self._axis_setter()
        # plot data
        if self.drawtype == "1":
            self._AI_data_setter()
        elif self.drawtype == "2":
            self._payoff_matrix_setter()
            self._reward_calculater_playerA()
        elif self.drawtype == "3":
            self._total_reward_calculator()
        else:
            exit(1)
        # save and show
        self._figure_out()

    def _axis_setter(self):
        # Figureと3DAxeS
        self.fig = plt.figure(figsize = (8, 8))
        self.ax = self.fig.add_subplot(111, projection="3d")
        # 軸ラベルを設定
        self.ax.set_xlabel("x", size = 16)
        self.ax.set_ylabel("y", size = 16)
        self.ax.set_zlabel("z", size = 16)
        # X axis represents cooperate rate with playerA.
        xmin, xmax = 0, 1
        # Y axis represents cooperate rate with playerB.
        ymin, ymax = 0, 1
        # Define the fineness of scale
        self.N = 41
        self.x = np.linspace(xmin, xmax, self.N)
        self.y = np.linspace(ymin, ymax, self.N)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.Z = np.zeros([self.N,self.N])
        
    def _payoff_matrix_setter(self):
        self.payoff_matrix= np.array([[(2,1),(-1,-1)], [(-1,-1),(1,2)]])
        if self.drawtype == "2":
            self.payoff_matrix = pad.normalize_payoff(payoff_matrix=self.payoff_matrix)


    def _AI_data_setter(self):
        # AI part
        testQN = QNetwork(0,True)
        actor = Actor()
        # This can't use numpy form calculation, therefore, we use manual calculation same as mesh.
        for m1 in range(self.N):
            for m2 in range(self.N):
                self.Z[m1][m2] = actor.get_action(np.array([self.x[m1],self.y[m2]]),np.inf,testQN)
        self._surface_drawer()
                
    def _reward_calculater_playerA(self):
        self.Z = pad.calculate_payoff(self.X,self.Y,self.payoff_matrix)
        self._surface_drawer()
        
    def _total_reward_calculator(self):
        #Total reward
        Z1 = 10*self.X*self.Y-5*self.X-5*self.Y+3
        Z2 = 5*self.X*self.Y -2*self.X-2*self.Y+1
        Z3 = 5*self.X*self.Y-3*self.X-3*self.Y+2
        self.ax.plot_surface(self.X,self.Y,Z2,facecolor = "#00F00060")
        self.ax.plot_surface(self.X,self.Y,Z1, facecolor = "#0000F010")
        self.ax.plot_surface(self.X,self.Y,Z3,facecolor = "#00F0F060")
        plt.plot(3/5,2/5,1/5,'*',color='r')
        self.ax.scatter(self.X,1-self.X,5*self.X*(1-self.X)-2*self.X-2*(1-self.X)+1,s = 2)
        # => self.ax.scatter(1-Y,Y,5*(1-Y)*Y-2*(1-Y)-2*Y+1,s=2)

           
        
    def _surface_drawer(self):
        minargZ = np.unravel_index(np.argmin(self.Z), self.Z.shape) 
        
        # 曲面を描画
        self.ax.plot_surface(self.X, self.Y, self.Z)
        plt.plot(self.X[minargZ[0]][0], \
            self.Y[minargZ[1]][0],\
            self.Z[minargZ],'*',color = "r",markersize=20)
        
        
    def _figure_out(self):
        plt.savefig("./data/figure3.png")
        with open("./data/figure3.pkl", "wb") as f:
            pickle.dump(self.fig, f)
        plt.show()

        








#ax.scatter(X,Y,Z)
# あとで使うために保存

if __name__ == "__main__":
    _ = payoff_matrix_concave_surface()