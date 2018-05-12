import numpy as np
from numpy.random import randn, rand
import argparse
import matplotlib.pyplot as pl
import seaborn as sns


def draw_GMM_parameters(d, K, seed = None):
    ''' randomly draws the parameters of the GMM'''
    if seed is not None:
        np.random.seed(seed)
    # weights
    w = rand(K)**2
    w /= np.sum(w)

    # means
    mu = randn(d, K)*K*10*np.log(d)

    # covariances
    L = (randn(d, d, K) * rand(1, 1, K) * K
         + np.eye(d)[..., None]*randn(1, 1, K)*np.sqrt(K))
    C = np.sum(L[..., None, :] * np.rollaxis(L, 1)[None, ...], axis=1)

    # compute inverses of covariances
    invC = np.zeros(C.shape)
    for k in range(K):
        invC[..., k] = np.linalg.pinv(C[..., k])

    return {'mu': mu, 'K': K, 'w': w, 'L': L, 'C': C, 'invC': invC}


def rand_GMM(params, T):
    ''' draw outcomes from a GMM '''
    # allocate output
    d = params['mu'].shape[0]
    x = np.zeros((T, d))
    y = np.zeros((T,))

    pos = 0
    for k in range(params['K']):
        # for each component, first pick the number of samples according to
        # the component weight
        Tc = min(T-pos, int(T*params['w'][k]))
        print(Tc)

        # draw the samples from this component
        xc = randn(d, Tc)
        xc = np.dot(params['L'][..., k], xc) + params['mu'][:, k][..., None]

        # aggregate to the output
        x[pos:pos+Tc] = xc.T
        y[pos:pos + Tc] = k
        pos += Tc

    order = np.random.permutation(T)
    x = x.flatten()[order] if d == 1 else x[order]
    y = y[order]
    return (x, y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
                                     'Generate some toy data')
    parser.add_argument("--output",
                        help="File to save samples to")
    parser.add_argument("--dim",
                        help="dimension of the samples",
                        type=int,
                        default=2)
    parser.add_argument("--num_samples",
                        help="Number of samples to draw",
                        type=int,
                        default=30000)
    parser.add_argument("--num_components",
                        help="Number of components in the GMM",
                        type=int,
                        default=30)
    parser.add_argument("--seed",
                        help="Seed to use for random generation of the GMM"
                             "parameters. If ommitted, will not use "
                             "a specific seed, yielding always different "
                             "data",
                        type=int)

    parser.add_argument("--plot",
                        help="Flag indicating whether or not to plot samples",
                        action="store_true")

    args = parser.parse_args()
    params = draw_GMM_parameters(args.dim, args.num_components, args.seed)
    (X, Y) = rand_GMM(params, args.num_samples)

    X = X-X.min()
    X /= X.max()
    #X *= 1000

    if args.plot:
        pl.plot(X[:, 0], X[:, 1], '.')
        pl.grid(True)
        pl.show()

    if args.output is not None:
        np.save(args.output, X)
