import matplotlib.pyplot as plt
import numpy 
import os

STOP_THRESHOLD = 0.001
MAX_ITERATION  = 1000
 
def mixture_gaussian(i):
    pi_0 = 0.3
    if numpy.random.random() < pi_0:
        return numpy.random.normal(-5, 1)
    else:
        return numpy.random.normal(5, 4)
 
mu = [-1.0, -1.0]
sigma = [1.0, 2.0]
pi = [0.5, 0.5]
gamma_0 = []
gamma_1 = []
sum_gamma = [0, 0]
N = 1000
x = [mixture_gaussian(i) for i in range(N)]



 
def norm_distribution(x, m, s, pi = 1.0):
    return pi * numpy.exp(-((x - m) ** 2) / (2 * s)) / numpy.sqrt(2 * numpy.pi * s)
 
 
def log_likelihood(x, m, s, pi):
    return sum([numpy.log(norm_distribution(x[i], m[0], s[0], pi[0]) + norm_distribution(x[i], m[1], s[1], pi[1]))
                for i in range(len(x))])
 

def main():

    # Getting the log likelyhood from the initial distribution
    new_log_likelihood = log_likelihood(x, mu, sigma, pi)
    data_len = len(x)

    # iteration

 
    for step in range(MAX_ITERATION):
        old_log_likelihood = new_log_likelihood
     
        for m_mixture in range(len(mu)):
            # E-step
            gamma = [norm_distribution(x[j], mu[m_mixture], sigma[m_mixture], pi[m_mixture]) /
                    sum([norm_distribution(x[j], mu[i], sigma[i], pi[i]) for i in range(2)])
                    for j in range(len(x))]

     
            # M-step
            sum_gamma = sum(gamma)
            mu[m_mixture] = sum([gamma[i] * x[i] / sum_gamma for i in range(data_len)])
            sigma[m_mixture] = sum([(gamma[i] * (x[i] - mu[m_mixture]) ** 2) / sum_gamma for i in range(data_len)])
         
            pi[m_mixture] = sum_gamma / N

        new_log_likelihood = log_likelihood(x, mu, sigma, pi)
        print "log likelyhood:\t" + str(old_log_likelihood) + '->' + str(new_log_likelihood)
        save_mixture_fig(x, mu, sigma, pi, step)
        if(abs(new_log_likelihood - old_log_likelihood) < STOP_THRESHOLD):
            break

    print mu
    print sigma
    print pi

    plot_mixture(x, mu, sigma, pi)


def plot_mixture(x, mu, sigma, pi):
    count, bins, ignored = plt.hist(x, 70, normed=True)
    for m_mixture in range(len(mu)):
        plt.plot(bins, norm_distribution(bins, mu[m_mixture], sigma[m_mixture], pi[m_mixture]), color='r')
    plt.show()


def save_mixture_fig(x, mu, sigma, pi, postfix_filename=0):
    count, bins, ignored = plt.hist(x, 70, normed=True)
    for m_mixture in range(len(mu)):
        plt.plot(bins, norm_distribution(bins, mu[m_mixture], sigma[m_mixture], pi[m_mixture]), color='r')
    plt.savefig("fig/fig_%2d.png" % postfix_filename)
    plt.close()

if __name__ == '__main__':
    os.system("rm -rf fig/*")
    main()