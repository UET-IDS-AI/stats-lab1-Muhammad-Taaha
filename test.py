import numpy as np
import matplotlib
matplotlib.use("Agg")

from stats_lab import (
    normal_histogram,
    uniform_histogram,
    bernoulli_histogram,
    sample_mean,
    sample_variance,
    order_statistics,
    sample_covariance,
    covariance_matrix
)

def run_test(name, func):
    try:
        func()
        print(f"✔ {name} PASSED")
        return True
    except AssertionError:
        print(f"❌ {name} FAILED")
        return False
    except Exception as e:
        print(f"❌ {name} ERROR: {e}")
        return False


# ---------------- TESTS ---------------- #

def test_q1_normal():
    np.random.seed(0)
    data = normal_histogram(10000)
    assert isinstance(data, (list, np.ndarray))
    data = np.asarray(data)
    assert abs(sample_mean(data)) < 0.1


def test_q1_uniform():
    np.random.seed(0)
    data = uniform_histogram(10000)
    data = np.asarray(data)
    assert abs(sample_mean(data) - 5) < 0.1


def test_q1_bernoulli():
    np.random.seed(0)
    data = bernoulli_histogram(10000)
    data = np.asarray(data)
    assert abs(sample_mean(data) - 0.5) < 0.05


def test_q2_mean():
    data = np.array([1,2,3,4,5])
    assert sample_mean(data) == 3


def test_q2_variance():
    data = np.array([1,2,3,4,5])
    assert np.isclose(sample_variance(data), 2.5)


def test_q3_order_statistics():
    data = np.array([5,1,3,2,4])
    minimum, maximum, median, q1, q3 = order_statistics(data)
    assert minimum == 1
    assert maximum == 5
    assert median == 3
    assert q1 == 2
    assert q3 == 4


def test_q4_covariance():
    x = np.array([1,2,3])
    y = np.array([2,4,6])
    assert np.isclose(sample_covariance(x,y), 2)


def test_q5_covariance_matrix():
    x = np.array([1,2,3])
    y = np.array([2,4,6])
    cm = covariance_matrix(x,y)
    assert cm.shape == (2,2)
    assert np.isclose(cm[0,1], 2)


# --------------- RUN ALL --------------- #

if __name__ == "__main__":
    
    tests = [
        ("Q1 Normal", test_q1_normal),
        ("Q1 Uniform", test_q1_uniform),
        ("Q1 Bernoulli", test_q1_bernoulli),
        ("Q2 Mean", test_q2_mean),
        ("Q2 Variance", test_q2_variance),
        ("Q3 Order Statistics", test_q3_order_statistics),
        ("Q4 Covariance", test_q4_covariance),
        ("Q5 Covariance Matrix", test_q5_covariance_matrix),
    ]
    
    passed = 0
    
    print("\nRunning Tests...\n")
    
    for name, test in tests:
        if run_test(name, test):
            passed += 1
    
    print("\n-----------------------------")
    print(f"Passed {passed} / {len(tests)} tests")
    
    if passed == len(tests):
        print("🎉 ALL TESTS PASSED SUCCESSFULLY 🎉")
    else:
        print("⚠ Some tests failed. Check above.")