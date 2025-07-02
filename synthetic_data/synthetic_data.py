

import unittest
import numpy as np
import random as rand
import math
import timeit

def gen_list(num_tokens: int, normalized_distr: any, error_function: any) -> np.ndarray:
    output: np.ndarray = np.zeros(num_tokens)
    for i in range(num_tokens):
        r = rand.normalvariate(0.5, 0.07)
        while abs(r - 0.5) > 0.5:
            r = rand.normalvariate(0.5, 0.07)
        (low, high) = error_function(normalized_distr(i))
        output[i] = (low ** (1 - r)) * (high ** r)
    output /= np.sum(output)
    return output

def check_lists_one_way(num_tokens: int, error_function: any, l1: np.ndarray, l2: np.ndarray) -> bool:
    for i in range(num_tokens):
        (low, high) = error_function(l1[i])
        # print(f'Element {i}. [{l1[i]}, {l2[i]}], [{low}, {high}]')
        if l2[i] <= low or l2[i] >= high:
            return False
    return True


def check_lists(num_tokens: int, error_function: any, l1: np.ndarray, l2: np.ndarray) -> bool:
    return check_lists_one_way(num_tokens, error_function, l1, l2) and check_lists_one_way(num_tokens, error_function, l2, l1)

def gen_normalized_distr(num_tokens: int, distr: any) -> np.ndarray:
    output = np.array([distr(i) for i in range(num_tokens)])
    sum_distr: int = np.sum(output)
    return output / sum_distr


def generate_synthetic_data(num_tokens: int, length: int, distr: any, error_function: any, estimated_accuracy = 0.9) -> tuple[np.ndarray, np.ndarray]:
    failures: int = 0
    sum_distr: int = sum(distr(i) for i in range(num_tokens))
    def normalized_distr(n):
        return distr(n) / sum_distr
    data_1: np.ndarray = np.zeros((length, num_tokens))
    data_2: np.ndarray = np.zeros((length, num_tokens))

    max_tries: int = math.ceil(1.1 * math.log(length * 100, 1 / (1 - estimated_accuracy)))

    print(max_tries)

    last_checked = timeit.default_timer()

    for i in range(length):
        if timeit.default_timer() - last_checked > 10:
            print(f'Working on iteration {i}')
            last_checked = timeit.default_timer()
        for _ in range(max_tries):
            l1: np.ndarray = gen_list(num_tokens, normalized_distr, error_function)
            l2: np.ndarray = gen_list(num_tokens, normalized_distr, error_function)
            if check_lists(num_tokens, error_function, l1, l2):
                perm = np.arange(num_tokens)
                rand.shuffle(perm)
                data_1[i] = np.take_along_axis(l1, perm, axis=0)
                data_2[i] = np.take_along_axis(l2, perm, axis=0)
                break
            failures += 1
        else:
            raise RuntimeWarning(f'Generation failed on iteration {i}.')
        
    print(f'Success rate: {100 * length / (failures + length):5.2f}%')
    return (data_1, data_2)

def new_var(x):
    y = x
    return y

class TestStringMethods(unittest.TestCase):


    
    power_law_distr = lambda n, ex: (n + 1) ** (ex)
    uniform_distr = lambda n: 1
    arctan_distr = lambda n: math.pi / 2 - math.atan(n)

    mult_error_func = lambda x, q: (x*q, x/q)
    arith_error_func = lambda x, diff: (max(0, x - diff), x + diff)
    
    def sine_error_func (x, k, smoothness, buffer): 
            return (
                -(k/smoothness) * abs(math.sin(x/k) + buffer) + x,
                (k/smoothness) * abs(math.sin(x/k) + buffer) + x
            )
    
        


    def test_listcheck_01(self):
        l1 = [0.5, 0.2, 0.1, 0.05, 0.05, 0.02, 0.02, 0.02, 0.02, 0.02]
        l2 = [0.5, 0.15, 0.15, 0.05, 0.05, 0.03, 0.02, 0.02, 0.015, 0.015]
        
        self.assertTrue(
            check_lists(10, lambda x: TestStringMethods.mult_error_func(x, 0.5), l1, l2)
        )
        self.assertTrue(
            check_lists(10, lambda x: TestStringMethods.mult_error_func(x, 1-2**(-4/3)), l1, l2)
        )

        self.assertFalse(
            check_lists(10, lambda x: TestStringMethods.mult_error_func(x, 1-2**(-5/3)), l1, l2)
        )

        l1 = [0.5, 0.1, 0.1, 0.1, 0.1, 0.1]
        l2 = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]

        self.assertFalse(
            check_lists_one_way(6, lambda x: (2/5 * x, 5 * x), l1, l2)
        )

        self.assertTrue(
            check_lists_one_way(6, lambda x: (2/5 * x, 5 * x), l2, l1)
        )



if __name__ == "__main__":
    # unittest.main()
    generate_synthetic_data(128256, 100, lambda n: (n + 1) ** -0.5, lambda x: (x/2, 2*x), 0.1)