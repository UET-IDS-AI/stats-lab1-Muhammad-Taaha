import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------
# Question 1 – Generate & Plot Histograms (and return data)
# -----------------------------------

def normal_histogram(n):
    """
    Generate n samples from Normal(0,1),
    plot a histogram with 10 bins (with labels + title),
    and return the generated data.
    """
    data = np.random.normal(0, 1, n)
    plt.hist(data, bins=10)
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.title("Normal Distribution")
    plt.show()
    return data
# normal_histogram(1000)

def uniform_histogram(n):
    """
    Generate n samples from Uniform(0,10),
    plot a histogram with 10 bins (with labels + title),
    and return the generated data.
    """
    data = np.random.uniform(0, 10, n)
    plt.hist(data, bins=10)
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.title("Uniform Distribution")
    plt.show()
    return data
# uniform_histogram(1000)


def bernoulli_histogram(n):
    """
    Generate n samples from Bernoulli(0.5),
    plot a histogram with 10 bins (with labels + title),
    and return the generated data.
    """
    data = np.random.binomial(1,0.5,n)
    plt.hist(data,bins=10)
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.title("Benoulli Histogram")
    plt.show()
    return data 
# bernoulli_histogram(1000)

# -----------------------------------
# Question 2 – Sample Mean & Variance
# -----------------------------------

def sample_mean(data):
    """
    Compute sample mean.
    """
    data = np.array(data)
    no_of_entries = len(data)
    sum_of_values= np.sum(data)
    avg = sum_of_values/no_of_entries
    return avg 
# ans=sample_mean([1,2,3,5,6,6,7,9,53,2,24,5,6,8])
# print(ans)


def sample_variance(data):
    """
    Compute sample variance using n-1 denominator.
    """
    mean = sample_mean(data)
    n = len(data)

    total = 0
    for value in data:
        total += (value - mean) ** 2

    return total / (n - 1)
# varience = sample_variance([1,2,3,3,4,4,56,7,8,8,9,1,2,2,33,4,4,5,66,777,888,11,22])
# print(varience)

# -----------------------------------
# Question 3 – Order Statistics
# -----------------------------------
def order_statistics(data):
    """
    Return:
    - min
    - max
    - median
    - 25th percentile (Q1)
    - 75th percentile (Q3)

    Use a consistent quartile definition. The tests for the fixed
    dataset [5,1,3,2,4] expect Q1=2 and Q3=4.
    """
    arr = np.array(data)
    arr = np.sort(arr)
    
    n = len(arr)
    
    minimum = arr[0]
    maximum = arr[-1]
    
    # Median
    if n % 2 == 0:
        median = (arr[n//2 - 1] + arr[n//2]) / 2
        lower_half = arr[:n//2]
        upper_half = arr[n//2:]
    else:
        median = arr[n//2]
        lower_half = arr[:n//2 + 1]
        upper_half = arr[n//2:]
    
    # Q1
    m = len(lower_half)
    if m % 2 == 0:
        q1 = (lower_half[m//2 - 1] + lower_half[m//2]) / 2
    else:
        q1 = lower_half[m//2]
    
    # Q3
    m = len(upper_half)
    if m % 2 == 0:
        q3 = (upper_half[m//2 - 1] + upper_half[m//2]) / 2
    else:
        q3 = upper_half[m//2]
    
    return (minimum, maximum, median, q1, q3)


# -----------------------------------
# Question 4 – Sample Covariance
# -----------------------------------

def sample_covariance(x, y):
    """
    Compute sample covariance using n-1 denominator.
    """
    n = len(x)

    mean_x = sample_mean(x)
    mean_y = sample_mean(y)

    total = 0
    for i in range(n):
        total += (x[i] - mean_x) * (y[i] - mean_y)

    return total / (n - 1)



# -----------------------------------
# Question 5 – Covariance Matrix
# -----------------------------------

def covariance_matrix(x, y):
    """
    Return 2x2 covariance matrix:
        [[var(x), cov(x,y)],
         [cov(x,y), var(y)]]
    """
    var_x = sample_variance(x)
    var_y = sample_variance(y)
    cov_xy = sample_covariance(x, y)

    return np.array([[var_x, cov_xy],
                     [cov_xy, var_y]])