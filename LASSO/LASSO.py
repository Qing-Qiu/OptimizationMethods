import numpy as np
import matplotlib.pyplot as plt
import scipy

seed = 114514
np.random.seed(seed)
m = 512
n = 1024
r = 0.1
A = np.random.randn(m, n)
u = scipy.sparse.random(n, 1, density=r, data_rvs=scipy.randn)
b = A * u
mu = 1e-3

Lambda = np.linalg.eig(np.matmul(A.T, A))
index = np.argmax(Lambda[0])
L = np.real(Lambda[0][index])
x0 = np.random.randn(n, 1)


class opts:
    def __init__(self):
        opts.maxit = 4000
        opts.ftol = 1e-8
        opts.alpha0 = 1 / L


class outs:
    def __init__(self):
        outs.fvec = []


def func(A, b, mu0, x):
    w = np.matmul(A, x) - b
    f = 0.5 * np.matmul(w.T, w) + mu0 * np.linalg.norm(x, 1)
    return f


def prox(x, mu):
    y = np.max(np.abs(x) - mu, 0)
    y = np.sign(x) * y
    return y


def LASSO_grad_huber(x, A, b, mu, mu0, opts):
    opts.maxit = 200
    opts.sigma = 0.1
    opts.alpha0 = 0.01
    opts.gtol = 1e-6
    opts.ftol = 1e-8
    r = np.matmul(A, x) - b
    g = np.matmul(A.T, r)

    huber_g = np.sign(x)
    idx = abs(x) < opts.sigma
    huber_g[idx] = x[idx] / opts.sigma

    g = g + mu * huber_g
    nrmG = np.linalg.norm(g, 2)
    f = .5 * np.linalg.norm(r, 2) ** 2 + mu * (
            np.sum(x[idx] ** 2 / (2 * opts.sigma)) + sum(abs(x[abs(x) >= opts.sigma]) - opts.sigma / 2))

    out = outs()
    out.fvec = .5 * np.linalg.norm(r, 2) ** 2 + mu0 * np.linalg.norm(x, 1)

    alpha = opts.alpha0
    eta = 0.2
    rhols = 1e-6
    gamma = 0.85
    Q = 1
    Cval = f
    for k in range(1, opts.maxit + 1):
        fp = f
        gp = g
        xp = x
        nls = 1
        while True:
            x = xp - alpha * gp
            r = np.matmul(A, x) - b
            g = np.matmul(A.T, r)
            huber_g = np.sign(x)
            idx = abs(x) < opts.sigma
            huber_g[idx] = x[idx] / opts.sigma
            f = .5 * np.linalg.norm(r, 2) ** 2 + mu * (
                    np.sum(x[idx] ** 2 / (2 * opts.sigma)) + sum(abs(x[abs(x) >= opts.sigma]) - opts.sigma / 2))
            g = g + mu * huber_g
            if f <= Cval - alpha * rhols * nrmG ** 2 or nls >= 10:
                break
            alpha = eta * alpha
            nls = nls + 1
        nrmG = np.linalg.norm(g, 2)
        forg = .5 * np.linalg.norm(r, 2) ** 2 + mu0 * np.linalg.norm(x, 1)
        out.fvec = np.append(out.fvec, forg)
        if nrmG < opts.gtol or abs(fp - f) < opts.ftol:
            break
        dx = x - xp
        dg = g - gp
        dxg = abs(np.matmul(dx.T, dg))
        if dxg > 0:
            if k % 2 == 0:
                alpha = np.matmul(dx.T, dx) / dxg
            else:
                alpha = dxg / (np.matmul(dg.T, dg))
            alpha = max(min(alpha, 1e12), 1e-12)
        Qp = Q
        Q = gamma * Qp + 1
        Cval = (gamma * Qp * Cval + f) / Q
    if k == opts.maxit:
        out.flag = 1
    else:
        out.flag = 0
    out.fval = f
    out.itr = k
    out.nrmG = nrmG
    return x, out


def LASSO_con(x0, A, b, mu0, opts):
    opts.maxit = 30
    opts.maxit_inn = 200
    opts.ftol = 1e-8
    opts.gtol = 1e-6
    opts.factor = 0.1
    opts.mu1 = 100
    opts.gtol_init_ratio = 1 / opts.gtol
    opts.ftol_init_ratio = 1e5
    opts.etaf = 1e-1
    opts.etag = 1e-1

    out = outs()
    out.fvec = []

    k = 0
    x = x0
    mu_t = opts.mu1
    f = func(A, b, mu_t, x)
    opts1 = opts()

    opts1.ftol = opts.ftol * opts.ftol_init_ratio
    opts1.gtol = opts.gtol * opts.gtol_init_ratio
    out.itr_inn = 0
    while k < opts.maxit:
        opts1.maxit = opts.maxit_inn
        opts1.gtol = max(opts1.gtol * opts.etag, opts.gtol)
        opts1.ftol = max(opts1.ftol * opts.etaf, opts.ftol)
        opts1.alpha0 = opts.alpha0
        opts1.sigma = 1e-3 * mu_t
        fp = f
        x, out1 = LASSO_grad_huber(x, A, b, mu_t, mu0, opts1)
        f = out1.fvec[-1]
        out.fvec = np.append(out.fvec, out1.fvec)
        k = k + 1
        nrmG = np.linalg.norm(x - prox(x - np.matmul(A.T, (np.matmul(A, x) - b)), mu0), 2)
        if ~out1.flag:
            mu_t = max(mu_t * opts.factor, mu0)
        if mu_t == mu0 and (nrmG < opts.gtol or abs(f - fp) < opts.ftol):
            break
        out.itr_inn = out.itr_inn + out1.itr
    out.fval = f
    out.itr = k
    return x, out


x, out = LASSO_con(x0, A, b, mu, opts)
f_star = min(out.fvec)

opts.maxit = 4000
x, out = LASSO_con(x0, A, b, mu, opts)

data1 = (out.fvec - f_star) / f_star
k1 = min(len(data1), 4000)
data1 = data1[1:k1]

mu = 1e-2
opts.ftol = 1e-8
opts.alpha0 = 1 / L
[x, out] = LASSO_con(x0, A, b, mu, opts)
f_star = min(out.fvec)

x, out = LASSO_con(x0, A, b, mu, opts)
data2 = (out.fvec - f_star) / f_star
k2 = min(len(data2), 4000)
data2 = data2[1:k2]

fig1 = plt.figure()
plt.semilogy(range(k1 - 1), data1, '-', color=[0.2, 0.1, 0.99], linewidth=2)
plt.semilogy(range(k2 - 1), data2, '-.', color=[0.99, 0.1, 0.2], linewidth=1.5)
plt.legend(['\u03BC = 10^-3', '\u03BC = 10^-2'])
plt.ylabel('$(f(x^k) - f^*)/f^*$', fontsize=14)
plt.xlabel('Iterations')
plt.savefig('1.png')

fig2 = plt.figure()
u = u.todense()
plt.subplot(2, 1, 1)
plt.plot(u, color=[0.2, 0.1, 0.99], marker='x', linestyle='none')
plt.xlim([1, 1024])
plt.title('Exact Solution')

plt.subplot(2, 1, 2)
plt.plot(x, color=[0.2, 0.1, 0.99], marker='x', linestyle='none')
plt.xlim([1, 1024])
plt.title('Gradient Descent Solution')

plt.savefig('2.png')
plt.show()
