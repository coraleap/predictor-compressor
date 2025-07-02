
import numpy as np
import unittest
import synthetic_data as synth

print(dir(synth))

def check_range(value: float, rng: tuple[float, float]) -> int:
    if (value >= rng[0]):
        return 1
    if (value < rng[1]):
        return -1
    return 0


def identify_bucket(value, buckets, min_ind = 0, max_ind = -1) -> tuple[int, tuple[float, float]]:
    if max_ind == -1:
        max_ind = len(buckets)
    if max_ind - min_ind == 0:
        raise RuntimeError('No Bucket Found!')
    if max_ind - min_ind == 1:
        return (min_ind, buckets[min_ind])

    middle_ind: int = (min_ind + max_ind) // 2

    rng: tuple[float, float] = buckets[middle_ind]
    comparison: int = check_range(value, rng)
    if comparison == 0:
        return (middle_ind, rng)
    if comparison == 1:
        return identify_bucket(
            value, buckets, min_ind, middle_ind
        )
    else:
        return identify_bucket(
            value, buckets, middle_ind + 1, max_ind
        )
    
def matching_prefix_length(s1: str, s2: str) -> int:
    output = 0
    while (len(s1) > output and len(s2) > output and s1[output] == s2[output]):
        output += 1
    return output

def invert_digit(digit: str, base: int) -> str:
    return str(
        (int(digit) + 1) % base
    )

def compress_single_token(
        token: int, pmf: np.ndarray, ref_table: list[str], 
        buckets: list[tuple[float, float]], error_range: any,
        bucket_prefixes: list[str], max_longform: int, base: int) -> str:
    bucket_index, target_bucket = identify_bucket(
        pmf[token], buckets
    )

    token_string = ref_table[token]
    expanded_bucket = (error_range(error_range(target_bucket[1])[0])[0], 
        error_range(
            error_range(target_bucket[0])[1])
            [1])
    
    other_strings = [ref_table[i] for i in range(len(pmf)) 
                     if check_range(pmf[i], expanded_bucket) and i != token
                     ]
    
    minimum_prefixfree_length = max(
        matching_prefix_length(token_string, s) for s in other_strings
    ) + 1

    if minimum_prefixfree_length + 1 >= max_longform:
        return bucket_prefixes[bucket_index] + token_string
    else:
        return bucket_prefixes[bucket_index] + token_string[:minimum_prefixfree_length] + invert_digit(
            token_string[minimum_prefixfree_length], base
        )

    
        

def create_package(pmf_generator: any, longform_strings: list[str] | dict[int, str], 
                   buckets_cutoffs: list[float] | list[tuple[float, float]], error_range: any,
                   bucket_prefixes: list[str], max_longform: int, base: int = 2
                   ) -> tuple[any, any]:
    ref_table = longform_strings
    if isinstance(longform_strings, dict):
        ref_table = [longform_strings[i] for i in range(len(longform_strings))]
    
    buckets = buckets_cutoffs
    if isinstance(buckets[0], int):
        buckets = zip(buckets_cutoffs, buckets_cutoffs[1:])

    F = error_range

    def compressor(inp: list[int]) -> str:
        pass
        
    def decompressor(inp: str) -> list[int]:
        pass
    
    return (compressor, decompressor)

class TestStringMethods(unittest.TestCase):
    def test_identify_bucket_01(self):
        test_buckets_1 = [[1, 0.5], [0.5, 0.3], [0.3, 0.2], [0.2, 0.1], [0.1, 0]]
        self.assertTrue(
            identify_bucket(0.6, test_buckets_1)[0] == 0
        )
        self.assertTrue(
            identify_bucket(0.5, test_buckets_1)[0] == 0
        )
        self.assertTrue(
            identify_bucket(0.3, test_buckets_1)[0] == 1
        )
        self.assertTrue(
            identify_bucket(0.2, test_buckets_1)[0] == 2
        )
    def test_identify_bucket_02(self):
        power_law_distr = lambda n, exp: (n + 1) ** (exp)
        test_distr_2 = [
            power_law_distr(n, -0.5) for n in range(100)
        ] + [0]
        test_distr_2 = list(zip(test_distr_2, test_distr_2[1:]))

        self.assertTrue(
            identify_bucket(2**(-0.5), test_distr_2)[0] == 0
        )
        self.assertTrue(
            identify_bucket(1/10, test_distr_2)[0] == 98
        )
        self.assertTrue(
            identify_bucket(1/241284, test_distr_2)[0] == 99
        )


if __name__ == "__main__":
    unittest.main()