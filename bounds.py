import torch
import numpy as np
import math
from scipy import optimize



def KL(Q, P):
    """
    Compute Kullback-Leibler (KL) divergence between distributions Q and P.
    """
    return sum([q * math.log(q / p) if q > 0.0 else 0.0 for q, p in zip(Q, P)])


def KL_binomial(q, p):
    """
    Compute the KL-divergence between two Bernoulli distributions of probability
    of success q and p. That is, Q=(q,1-q), P=(p,1-p).
    """
    return KL([q, 1.0 - q], [p, 1.0 - p])


def solve_kl_sup(q, right_hand_side):
    """
    find x such that:
        kl( q || x ) = right_hand_side
        x > q
    """

    f = lambda x: KL_binomial(q, x) - right_hand_side
    if f(1.0 - 1e-9) <= 0.0:
        return 1.0 - 1e-9
    else:
        return optimize.brentq(f, q, 1.0 - 1e-11)



def solve_kl_inf(q, right_hand_side):
    """
    find x such that:
        kl( q || x ) = right_hand_side
        x < q
    """
 

    f = lambda x: KL_binomial(q, x) - right_hand_side
    if f(1e-9) <= 0.0:
        return 1e-9
    else:
        return optimize.brentq(f, 1e-11, q)



def get_loss(input, target, limit):
    loss = (input - target) ** 2
    loss = torch.clamp(loss, max=limit)
    # emploss = loss / loss.max().item()
    return loss


def get_kl(mu1, var1, mu2, var2):
    return var2.log() - var1.log() + (var1 + (mu1 - mu2) ** 2) / (2 * var2) - 0.5

def kl_div_gaussian(eval_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kl_values = []
    for i, batch in enumerate(eval_loader):
        kl_ = batch[0][5]
        kl_values.append(kl_.item())
    print("KL Divergence values per step:", kl_values)
    return kl_values


################# compute pac-bayes loss terms seprately #################
def mcsampling_excess(input_p, target_p, input, target, limit, gamma_t=0.5):
    """Compute the mean of excess loss delta_j^{\hat}(h_2, h_1, X, Y), where"""
    loss_prior = 0
    loss_posterior = 0
    loss_prior = get_loss(input_p, target_p, limit)
    loss_posterior = get_loss(input, target, limit)
    delta = get_excess(loss_posterior, loss_prior, gamma_t)
    n_ = abs(loss_prior.shape[0] - loss_posterior.shape[0])
    posteriorloss = loss_posterior.mean().item()
    priorloss = loss_prior.mean().item()
    

    ex_emp = loss_posterior.mean() - gamma_t * (loss_prior[n_:]).mean()
    if math.isclose(ex_emp.item(), delta.mean().item(), rel_tol=1e-3):
        print("loss_excess avg is equal to (avg posteriorloss - gamma * avg priorloss)")
    else:
        print( f"ecxess_of_avg: {ex_emp.item()},excess of all: {delta.mean().item()}")

    return delta.cpu().numpy(), posteriorloss, priorloss


def get_excess(loss_posterior, loss_prior, gamma_t):
    """Compute excess loss delta_j^{\hat}(h_2, h_1, x, y) for a data and each j."""
    # added to handel cases where the episode length is not 1000
    if loss_prior.shape != loss_posterior.shape:
        n = abs(loss_prior.shape[0] - loss_posterior.shape[0])
        if loss_prior.shape[0] > loss_posterior.shape[0]:
            loss_prior = loss_prior[n:]
        else:
            raise ValueError("loss_posterior is greater than loss_prior")
        
    delta = loss_posterior - gamma_t * loss_prior
    return delta 



def compute_E_t(loss_excess, kl, T, gamma_t, n_bound, B, delta_test=0.01, delta=0.025):
    mu = 0
    inv_1 = solve_kl_sup((torch.maximum(torch.tensor(0.0), loss_excess - mu)).mean().item(), np.log(2 * T / delta_test) / n_bound)
    inv_2 = solve_kl_sup(inv_1 / (B - mu), (kl + np.log(( T * 4 * np.sqrt(n_bound)) / delta)) / (n_bound))
    inv_1_inf = solve_kl_inf((torch.maximum(torch.tensor(0.0), mu - loss_excess)).mean().item(), np.log(2 * T / delta_test) / n_bound)
    inv_2_inf = solve_kl_inf(inv_1_inf / (mu + gamma_t*B), (kl + np.log(( T * 4 * np.sqrt(n_bound)) / delta)) / (n_bound))
    
    E_t = mu + ((B - mu) * inv_2) - ((mu + gamma_t*B) * inv_2_inf)
    return E_t, inv_1/(B - mu), inv_2, inv_1_inf/(mu + gamma_t*B), inv_2_inf, (kl + np.log(( T * 4 * np.sqrt(n_bound)) / delta)) / (n_bound)

def compute_B_1(emp_loss, kl, T, n_bound, B, delta_test=0.01, delta=0.025, gamma_t = 0.5):
    mu = 0
    inv_1_sup = solve_kl_sup((torch.maximum(torch.tensor(0.0), emp_loss - mu)).mean().item(), np.log(T / delta_test) / n_bound)
    kl_sup = solve_kl_sup(inv_1_sup / B, (kl + np.log((4 * T * np.sqrt(n_bound)) / delta)) / (n_bound))
    inv_1_inf = solve_kl_inf((torch.maximum(torch.tensor(0.0), mu - emp_loss)).mean().item(), np.log(T / delta_test) / n_bound)
    kl_inf = solve_kl_inf(inv_1_inf / (gamma_t*B), (kl + np.log((4 * T * np.sqrt(n_bound)) / delta)) / (n_bound))
    return (B * kl_sup) - (gamma_t * B * kl_inf), inv_1_sup, B * kl_sup

def compute_B_t(B_1, E_ts, gamma_t):
    """Compute risk of T-step posteriors using the recursive formula:
    B_t = E_t + gam * B_{t-1}
    """
    B_ts = [B_1]
    for i in range(len(E_ts)):
        B_t = B_ts[i] * gamma_t + E_ts[i]
        B_ts.append(B_t)
    return B_ts

def compute_inf_risk_invkl(eval_data_list, delta_test=0.01, delta=0.025):
    n_bound = len(eval_data_list)
    emploss = torch.cat([e[6] for e in eval_data_list])
    B, kl = eval_data_list[0][3], eval_data_list[0][4].item()
    
    inv_1 = solve_kl_sup(emploss.mean().item(), np.log(1 / delta_test) / n_bound)
    risk = solve_kl_sup(inv_1 / B, (kl + np.log((2 * np.sqrt(n_bound)) / delta)) / n_bound)
    
    return {"emploss": emploss.mean().item(), "fullloss": B * risk, "rightside": (kl + math.log(2.0*math.sqrt(n_bound)/delta))/n_bound}

########################### compute recursive pac-bayes bound ##################################
def compute_risk_rpb(eval_loaders, gamma_t=0.5, delta_test=0.01, delta=0.025):
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = len(eval_loaders)

    # Precompute KL divergences for all steps
    kl_ts = kl_div_gaussian(eval_loaders)

    # Initialize variables
    loss_ts = []
    E_ts = []
    E=[]
    B_i=[]
    max_B = []
    bound_size = []
    posterioremploss = []
       
    # Iterate through each step
    for t in range(1, T + 1):
        n_bound = len(eval_loaders[t - 1])
        print(f"Current step: {t}", f"len of bound: {n_bound}")
        # KL divergence for the current step
        kl = kl_ts[t - 1]
        upper_limit = eval_loaders[t - 1][0][4]  # B

        if t == 1:
            risk = torch.cat([element[7] for element in eval_loaders[t - 1]]) # emploss
            loss = risk

            B_1,_,_ = compute_B_1(loss, kl, T, n_bound, upper_limit,  delta_test, delta, gamma_t)
            B_i.append(B_1)
            max_B.append(upper_limit)
            E.append((0, 0, 0, 0, 0, 0))
            loss_ts.append(0)
            posterioremploss.append(loss.mean().item())
        else:
            # Compute (E_t)_{t >= 1}
            target_prior = torch.cat([element[1] for element in eval_loaders[t - 2]])
            predict_prior = torch.cat([element[2] for element in eval_loaders[t - 2]])
            target = torch.cat([element[1] for element in eval_loaders[t - 1]])
            predict = torch.cat([element[2] for element in eval_loaders[t - 1]])
           
            upper_limit = ([element[4] for element in eval_loaders[t - 1]])[0]
            max_B_new = upper_limit

            emp_losses = mcsampling_excess(
                predict_prior,
                target_prior,
                predict,
                target,
                upper_limit,
                gamma_t=gamma_t,
            )
            loss_excess = emp_losses[0]
            loss_ts.append(emp_losses[0].mean().item())
            posterioremploss.append(emp_losses[1])

            E_t_, inv_b , Epsilon_plas, invinf_gammab , Epsilon_neg, kl_n = compute_E_t(
                torch.tensor(loss_excess), kl, T, gamma_t, n_bound, max_B_new, delta_test, delta
            )
            if t == T:
                B_T_inv = compute_risk_last_invkl( eval_loaders[t - 1], kl, max_B_new, delta_test, delta)
            E_ts.append(E_t_)
            E.append((E_t_, inv_b, Epsilon_plas, invinf_gammab, Epsilon_neg, kl_n))
            B_i.append(B_i[-1] * gamma_t + E_t_)
            max_B.append(max_B_new)
        bound_size.append(n_bound)

    # Compute B_t recursively using B_1, (E_t)_t, and gamma_t
    B_ts = compute_B_t(B_1, E_ts, gamma_t)
    print(f"B_ts_list_all steps: {B_ts}, Recursive pac-bayes bound: {B_ts[-1]}")
    print(f"excess loss: {loss_ts}")
    B_T_MC = 0

    return loss_ts, kl_ts, E, B_ts, B_T_MC, max_B , B_T_inv, bound_size, posterioremploss


##################### informative inv-kl bound ###########################################
# def compute_inf_risk_invkl(eval_loader, delta_test=0.01, delta=0.025):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     n_bound = len(eval_loader)
#     target = torch.cat([element[1] for element in eval_loader])
#     predict = torch.cat([element[2] for element in eval_loader])
#     emploss = torch.cat([element[6] for element in eval_loader]) # emploss
#     B, kl = eval_loader[0][3], eval_loader[0][4].item() # B, kl

#     N = n_bound
#     last_kl_inv = {
#     "emploss": 0,
#     "rightside": 0,
#     "fullloss": 0,
#     "leftside": 0
#     }
#     risk = 0

#     inv_1 = solve_kl_sup(emploss.mean().item(), np.log(1 / delta_test) / n_bound)
#     risk = solve_kl_sup(
#         inv_1 / B,
#         (kl + np.log((2 * np.sqrt(n_bound)) / delta)) / n_bound,
#     )
#     confidence_term = math.log(2.0 * math.sqrt(N) / delta) 
#     rightside = ((kl+confidence_term) / N) 
#     loss = B * risk

#     last_kl_inv["emploss"] = emploss.mean().item()
#     last_kl_inv["rightside"] = rightside
#     last_kl_inv["fullloss"] = loss
#     last_kl_inv["leftside"] = inv_1 / B
#     return last_kl_inv
#####################################################################################
def compute_risk_last_invkl( eval_loader,kl,B, delta_test, delta):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = len(eval_loader)
    target = torch.cat([element[1] for element in eval_loader])
    predict = torch.cat([element[2] for element in eval_loader])
    emploss = torch.clamp(
            (predict - target) ** 2, max = B ).mean()

    loss = 0
    n_bound = N
  
    last_kl_inv = {
    "emploss": 0,
    "rightside": 0,
    "fullloss": 0,
    "leftside": 0
    }
    risk = 0
    
    inv_1 = solve_kl_sup(emploss.mean().item(), np.log(1 / delta_test) / n_bound)
    risk = solve_kl_sup(
        inv_1 / B,
        (kl + np.log((2 * np.sqrt(n_bound)) / delta)) / n_bound,
    )
    confidence_term = math.log(2.0 * math.sqrt(N) / delta) 
    rightside = ((kl+confidence_term) / N) 
    loss = B * risk

    last_kl_inv["emploss"] = emploss.mean().item()
    last_kl_inv["rightside"] = rightside
    last_kl_inv["fullloss"] = loss
    last_kl_inv["leftside"] = inv_1 / B
    return last_kl_inv