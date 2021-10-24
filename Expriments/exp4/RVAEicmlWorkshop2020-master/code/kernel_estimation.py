from numpy import array, linspace

from sklearn.neighbors.kde import KernelDensity

from matplotlib.pyplot import plot

a = array([10,11,9,23,21,11,45,20,11,12]).reshape(-1, 1)

kde = KernelDensity(kernel='gaussian', bandwidth=3).fit(a)

s = linspace(0,50)

e = kde.score_samples(s.reshape(-1,1))

plot(s, e)