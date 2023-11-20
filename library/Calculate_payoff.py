import numpy as np

class Payoff_dealer:
    def flatten_payoff_matrix(payoff_matrix):
        flatten_payoff_matrix = np.ravel(payoff_matrix)
        return (flatten_payoff_matrix)

    def normalize_payoff(payoff_matrix):
        payoff_matrix = payoff_matrix / np.linalg.norm(payoff_matrix) # normalization
        return payoff_matrix


    def calculate_payoff(action1, action2 ,payoff:np.array):

        payoff_r = payoff[0,0,0]*action1*action2+ \
        payoff[0,1,0]*action1*(1-action2)+\
        payoff[1,0,0]*(1-action1)*action2+\
        payoff[1,1,0]*(1-action1)*(1-action2)
        
        invalid_actions = np.where(action1>1);np.put(payoff_r,invalid_actions,-0.5)
        invalid_actions = np.where(action1<0);np.put(payoff_r,invalid_actions,-0.5)
    
        return payoff_r