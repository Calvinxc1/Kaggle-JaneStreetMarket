import numpy as np
from tqdm.notebook import tqdm

def loss_calc(val, dist_one, dist_two):
    p_one = dist_one.cdf(val)
    p_two = 1-dist_two.cdf(val)
    loss = (p_one - p_two)**2
    return loss

def prob_calc(top_val, bottom_val, temp):
    prob = np.exp(((top_val/bottom_val) - 1) / temp)
    return prob

def anneal(dist_one, dist_two, sample_dist, shape=0.95, scale=100, iters=1000, loss_calc=loss_calc, prob_calc=prob_calc, verbose=False):
    best_val = sample_dist.rvs()
    best_loss = loss_calc(best_val, dist_one, dist_two)

    temp_vals = scale * (shape ** np.arange(iters))
    t = tqdm(temp_vals) if verbose else temp_vals
    for temp in t:
        val = sample_dist.rvs()
        loss = loss_calc(val, dist_one, dist_two)
        prob = prob_calc(best_loss, loss, temp)
        if np.isinf(prob):
            best_val = val
            best_loss = loss
        elif np.random.uniform() < prob:
            best_val = val
            best_loss = loss

        if verbose: t.set_postfix(dict(loss=best_loss, val=best_val, temp=temp))
            
    return best_val, best_loss
