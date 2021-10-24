import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
import torch
import math
from torch.autograd import Variable

# https://github.com/mlosch/pytorch-stats/tree/master/stats/distributions
import torch
import numpy as np
import numbers
from torch.autograd import Variable
from matplotlib.widgets import Cursor

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"

class Normal(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, *args):
        x = args[0]
        mean = self.mean.expand(x.size())
        var = self.std.expand(x.size()) ** 2
        p = 1. / torch.sqrt(2.0 * np.pi * var) * torch.exp(- ((x - mean) ** 2) / (2.0 * var)) + 0.000001
        return p


def tensor(arr, dtype=torch.DoubleTensor):
    """
    Converts a float or an (numpy) array to a torch tensor.
    Parameters
    ----------
    arr :   float or list or ndarray
            Scalar or array of floats
    dtype : torch.dtype
    Returns
    -------
    Torch Tensor
    """
    if isinstance(arr, numbers.Number):
        t = torch.ones(1) * arr
    elif type(arr) is list or type(arr) is tuple:
        t = torch.Tensor(arr)
    else:
        t = torch.from_numpy(np.array(arr))
    return t.type(dtype)


def gen_contaminated_data(true_mean=-10, true_std=5, size_orig=100000,
                          outlier_percentage=10, outlier_mean=150, outlier_std=5):
    x = true_mean + np.random.randn(size_orig) * true_std
    # x = np.random.gamma(true_mean, true_std, size_orig)
    outliers = outlier_mean + np.random.randn(int(outlier_percentage * size_orig / 100)) * outlier_std
    x_all = np.concatenate([x, outliers])
    x_all = Variable(tensor(x_all))
    return x_all


# def p(x, start=-30, end=100, n_samples=1000):
#     x_tmp = np.linspace(start, end, n_samples)
#     counts, bins = np.histogram(x.data.numpy(), bins=x_tmp, density=True)
#     counts /= sum(counts)
#     bin_indices = np.digitize(x.data.numpy(), bins)
#     epsilon = 0.00001
#     return torch.tensor(counts[bin_indices] + epsilon)

def p(x_perfect, observations):
    counts, bins = np.histogram(observations.data.numpy(), bins=x_perfect, density=True)
    counts /= sum(counts)
    bin_indices = np.digitize(x_perfect.data.numpy(), bins)
    bin_indices = bin_indices[bin_indices < len(counts)]
    epsilon = 1e-12
    return torch.tensor(counts[bin_indices] + epsilon)

def KL(p, q):
    epsilon = 1e-12
    p = p + epsilon
    q = q + epsilon
    size = min(len(p), len(q))
    p = p[:size]
    q = q[:size]
    divergence = torch.sum(p*torch.log(p/q))
    return divergence


def rKL(p, q):
    return KL(q, p)


def gamma_div(p, q, gamma=0.5):
    epsilon = 1e-12
    p = p + epsilon
    q = q + epsilon
    size = min(len(p), len(q))
    p = p[:size]
    q = q[:size]
    term1 = 1/(gamma*(gamma + 1))*torch.log(torch.sum(p**(gamma + 1)))
    term2 = 1/(gamma+1)*torch.log(torch.sum(q**(gamma + 1)))
    term3 = 1/gamma*torch.log(torch.sum(p*(q**gamma)))
    return term1+term2-term3


def beta_div(p, q, beta=0.5):
    epsilon = 1e-12
    p = p + epsilon
    q = q + epsilon
    size = min(len(p), len(q))
    p = p[:size]
    q = q[:size]
    term1 = (1/beta)*(torch.sum(p**(beta + 1)))
    term2 = (torch.sum(q**(beta + 1)))
    term3 = ((beta+1)/beta)*(torch.sum(p*(q**beta)))
    return (term1+term2-term3)


def JS(p, q):
    size = min(len(p), len(q))
    p = p[:size]
    q = q[:size]
    divergence = 0.5*KL(p, (p+q)/2) + 0.5*KL(q, (p+q)/2)
    return divergence


def TV(p, q):
    epsilon = 1e-12
    p = p + epsilon
    q = q + epsilon
    size = min(len(p), len(q))
    p = p[:size]
    q = q[:size]
    abs_diff = torch.abs(p-q)
    index = torch.argmax(abs_diff)
    return abs_diff[int(index)]

plt.close('all')
# parameters
iterations = 10000
lr = 0.5
beta_list = [0.01, 0.1, 0.3, 0.5]
divergence_list = ['KL', 'beta-divergence']
gamma = 0.5

x = gen_contaminated_data(true_mean=10, true_std=5, size_orig=100000,
                          outlier_percentage=10, outlier_mean=150, outlier_std=5)
x_perfect = torch.tensor(np.linspace(-200, 200, 500))
p1 = p(x_perfect, x) / torch.sum(p(x_perfect, x))

# divergence_list = ['KL', 'beta-divergence'] #['KL', 'reversed-KL', 'Jensen-Shannon', 'TV', 'beta-divergence', 'gamma-divergence']

for divergence in divergence_list:
    for beta in beta_list:

        if divergence == 'KL':
            loss = KL
            kargs = {}
        elif divergence == 'reversed-KL':
            loss = rKL
            kargs = {}
        elif divergence == 'Jensen-Shannon':
            loss = JS
            kargs = {}
        elif divergence == 'beta-divergence':
            loss = beta_div
            kargs = {'beta': beta}
        elif divergence == 'gamma-divergence':
            loss = gamma_div
            kargs = {'gamma': gamma}
        elif divergence == 'TV':
            loss = TV
            kargs = {}

        mean = Variable(tensor(100), requires_grad=True)
        std = Variable(tensor(100), requires_grad=True)
        q = Normal(mean, std)

        parameters = [mean, std]
        optimizer = torch.optim.Adam(parameters, lr=0.01)

        likelihood_list = []
        for i in range(iterations):
            if i%100 == 0:
                print('iter=%d, mean=%g, std=%g'%(i,mean,std))

            p2 = q(x_perfect)/torch.sum(q(x_perfect))
            optimizer.zero_grad()
            likelihood = loss(p1, p2, **kargs)
            likelihood_list.append(float(likelihood))
            # Determine gradients
            likelihood.backward()
            # # Update parameters with gradient descent
            # for param in parameters:
            #     param.data.sub_(lr * param.grad.data)
            #     param.grad.data.zero_()
            optimizer.step()

        mean_tmp, var_tmp = float(parameters[0]), float(parameters[1]) ** 2
        x_tmp = np.linspace(-200, 200, 500)  # observations.data.numpy()
        y = 1. / np.sqrt(2.0 * np.pi * var_tmp) * np.exp(- ((x_tmp - mean_tmp) ** 2) / (2.0 * var_tmp))

        if divergence == 'KL':
            y_KL = y
        elif divergence == 'beta-divergence' and beta == 0.1:
            y_beta_0dot1 = y
        elif divergence == 'beta-divergence' and beta == 0.3:
            y_beta_0dot3 = y
        elif divergence == 'beta-divergence' and beta == 0.5:
            y_beta_0dot5 = y
        elif divergence == 'beta-divergence' and beta == 0.01:
            y_beta_0dot01 = y
        # plt.show()

        plt.figure()
        counts, bins = np.histogram(x.data.numpy(), bins=x_tmp, density=True)
        plt.plot(bins[:-1], counts / sum(counts), color='red', label='Original Distribution')
        plt.plot(x_tmp, y, color='blue', linestyle='dashed', label='Estimated Distribution')
        # plt.semilogy()
        plt.xlabel('x')
        title = divergence
        if divergence == 'beta-divergence':
            title += "-beta=" + str(beta)
        if divergence == 'gamma-divergence':
            title += "-gamma=" + str(gamma)
        plt.title(title)
        plt.legend(loc='upper right')
        plt.savefig('./results/' + title + '.png')

        plt.figure()
        plt.plot((np.array(likelihood_list)))
        plt.ylabel('Loss')
        plt.xlabel('iterations')
        plt.title(title)
        plt.savefig('./results/simulations_loss_' + title + '.png')

        if divergence == 'KL':
            break

plt.figure()
plt.plot(bins[:-1], counts / sum(counts), color='green', label='Original', linewidth=2)
plt.plot(x_tmp, y_KL, label='Estimated via KL divergence', color='red', linestyle='dashdot', linewidth=2)
# plt.plot(x_tmp, y_beta_0dot01, label=r'Estimated via $\beta=0.01$ divergence', linewidth=2)
plt.plot(x_tmp, y_beta_0dot1, label=r'Estimated via $\beta$ divergence', color='blue', linestyle='dashed', linewidth=2)

plt.annotate("Normal data", xy=(12,0.059), xytext=(-80,20),textcoords="offset points",
                    #bbox=dict(boxstyle='round4', fc='linen',ec='k',lw=1),
                    arrowprops=dict(arrowstyle='-|>', facecolor='black', color = 'black', edgecolor = 'black'),
                    horizontalalignment='left', verticalalignment='top')
plt.annotate("Outliers", xy=(147,0.007), xytext=(-25,20),textcoords="offset points",
                    #bbox=dict(boxstyle='round4', fc='linen',ec='k',lw=1),
                    arrowprops=dict(arrowstyle='-|>', facecolor='black', color = 'black', edgecolor = 'black'))
# annot.set_visible(True)
# plt.plot(x_tmp, y_beta_0dot3, label=r'Estimated via $\beta=0.3$ divergence', linewidth=2)
# plt.plot(x_tmp, y_beta_0dot5, label=r'Estimated via $\beta=0.5$ divergence', linewidth=2)
plt.legend(fontsize=8)
plt.grid(True)
plt.xlabel(r'$x$', fontsize=10)
plt.ylabel(r'$p(x)$', fontsize=10)
plt.savefig('./results/simulations_compare_KL_with_beta1.pdf')