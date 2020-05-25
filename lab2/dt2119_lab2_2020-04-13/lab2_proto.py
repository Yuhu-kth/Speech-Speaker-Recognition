import numpy as np
from lab2_tools import *

def concatTwoHMMs(hmm1, hmm2):
    """ Concatenates 2 HMM models

    Args:
       hmm1, hmm2: two dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be different for each)

    Output
       dictionary with the same keys as the input but concatenated models:
          startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models
   
    Example:
       twoHMMs = concatHMMs(phoneHMMs['sil'], phoneHMMs['ow'])

    See also: the concatenating_hmms.pdf document in the lab package
    """
    name = hmm1['name']+hmm2['name']

    startprob1 = np.array(hmm1["startprob"])
    startprob2 = np.array(hmm2["startprob"])
    startprob = np.hstack((startprob1[0:-1],startprob1[-1]*startprob2))

    transmat1 = hmm1["transmat"]
    transmat2 = hmm2["transmat"]
    part1 = np.hstack((transmat1[0:-1,0:-1],np.outer(transmat1[:-1,-1],startprob2)))
    part2 = np.hstack((np.zeros((transmat2.shape[0],transmat1.shape[1]-1)),transmat2))
    transmat = np.vstack((part1,part2))
    
    means1 = hmm1["means"]
    means2 = hmm2["means"]
    means = np.vstack((means1,means2))

    covars1 = hmm1["covars"]
    covars2 = hmm2["covars"]
    covars = np.vstack((covars1,covars2))

    dict_ = {'name':name,'startprob':startprob,'transmat':transmat,'means':means,'covars':covars}
    return dict_
    

# this is already implemented, but based on concat2HMMs() above
def concatHMMs(hmmmodels, namelist):
    """ Concatenates HMM models in a left to right manner

    Args:
       hmmmodels: dictionary of models indexed by model name. 
       hmmmodels[name] is a dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances
       namelist: list of model names that we want to concatenate

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be
      different in each model)

    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models:
         startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models

    Example:
       wordHMMs['o'] = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])
    """
    concat = hmmmodels[namelist[0]]
    for idx in range(1,len(namelist)):
        concat = concatTwoHMMs(concat, hmmmodels[namelist[idx]])
    return concat


def gmmloglik(log_emlik, weights):
    """Log Likelihood for a GMM model based on Multivariate Normal Distribution.

    Args:
        log_emlik: array like, shape (N, K).
            contains the log likelihoods for each of N observations and
            each of K distributions
        weights:   weight vector for the K components in the mixture

    Output:
        gmmloglik: scalar, log likelihood of data given the GMM model.
    """

def forward(log_emlik, log_startprob, log_transmat):
    """Forward (alpha) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i M
        log_transmat: log transition probability from state i to j MxM

    Output:
        forward_prob: NxM array of forward log probabilities for each of the M states in the model
    """
    logPi=log_startprob[:-1]
    logB=log_emlik
    logA=log_transmat[:-1,:-1]
    alpha = np.zeros_like(logB)
    alpha[0]=logB[0]+logPi
    for i in range(1,logB.shape[0]):
        alpha[i]=logsumexp(alpha[i-1]+logA.T+logB[i].reshape(-1,1),axis=1)
    return alpha


def backward(log_emlik, log_startprob, log_transmat):
    """Backward (beta) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        backward_prob: NxM array of backward log probabilities for each of the M states in the model
    """
    N, M = log_emlik.shape
    logPi=log_startprob[:-1]
    logB=log_emlik
    logA=log_transmat[:-1,:-1]
    beta = np.zeros_like(logB)
    for t in range(N-2,-1,-1):
        beta[t]=logsumexp(beta[t+1]+logA+logB[t+1].reshape(-1,1),axis=1)

    return beta

def viterbi(log_emlik, log_startprob, log_transmat, forceFinalState=True):
    """Viterbi path.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j
        forceFinalState: if True, start backtracking from the final state in
                  the model, instead of the best state at the last time step

    Output:
        viterbi_loglik: log likelihood of the best path
        viterbi_path: best path
    """
    N, M = log_emlik.shape
    logPi=log_startprob[:-1]
    logB=log_emlik
    logA=log_transmat[:-1,:-1]
    theta = np.zeros_like(logB)
    prev = np.zeros(logB.shape,dtype=np.int64)
    path = np.zeros(N,dtype=np.int64)
    theta[0] = logPi + logB[0]
    for t in range(1,N):
        temp = theta[t-1]+logA.T+logB[t].reshape(-1,1)
        theta[t]=np.max(temp,axis=1)
        prev[t]=np.argmax(temp,axis=1)
    if forceFinalState:
        path[-1] = M-1
    else:
        path[-1] = np.argmax(prev[-1])
    for i in range(N-2,-1,-1):
        path[i] = prev[i+1][path[i+1]]
    return np.max(theta[-1]),path

def statePosteriors(log_alpha, log_beta):
    """State posterior (gamma) probabilities in log domain.

    Args:
        log_alpha: NxM array of log forward (alpha) probabilities
        log_beta: NxM array of log backward (beta) probabilities
    where N is the number of frames, and M the number of states

    Output:
        log_gamma: NxM array of gamma probabilities for each of the M states in the model
    """
    return log_alpha + log_beta - logsumexp(log_alpha[-1,:])

def updateMeanAndVar(X, log_gamma, varianceFloor=5.0):
    """ Update Gaussian parameters with diagonal covariance

    Args:
         X: NxD array of feature vectors
         log_gamma: NxM state posterior probabilities in log domain
         varianceFloor: minimum allowed variance scalar
    were N is the lenght of the observation sequence, D is the
    dimensionality of the feature vectors and M is the number of
    states in the model

    Outputs:
         means: MxD mean vectors for each state
         covars: MxD covariance (variance) vectors for each state
    """
    D = X.shape[1]
    N, M = log_gamma.shape
    means = np.zeros((M, D))
    covars = np.zeros((M, D))
    
    gamma = np.exp(log_gamma)

    means = gamma.T[:,:,np.newaxis]*X[np.newaxis,:,:] # NxMxD
    means = means .sum(1)
    means = means/ gamma.sum(0).T[:,np.newaxis]
    
    X_mu = X[:,np.newaxis,:]-means[np.newaxis,:,:]# NxMxD
    X_mu_2 = X_mu *X_mu
    covars = (X_mu_2*gamma[:,:,np.newaxis]).sum(0)/ gamma.sum(0).T[:,np.newaxis]
    covars[covars < varianceFloor] = varianceFloor
    return means,covars