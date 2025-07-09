
import numpy as np
import unittest
try:
    from synthetic_data import generate_synthetic_data
except:
    from .synthetic_data import generate_synthetic_data
import math

"""
 ---------- HELPERS ----------
"""

def check_range(value: float, rng: tuple[float, float]) -> int:
    """
    Checks if a value v is within a given range (a, b] where a > b. 
    If v is above this range, returns 1. 
    If v is in this range, returns 0.
    If v is below this range, returns -1.

    Examples:
    value: 2
    rng: [2, 1]
    -> False

    value: 1
    rng: [2, 1]
    -> True
    """
    if (value >= rng[0]):
        return 1
    if (value < rng[1]):
        return -1
    return 0


def identify_bucket(value: float, buckets: list[tuple[float, float]], 
    min_ind: int = 0, max_ind: int = -1) -> tuple[int, tuple[float, float]]:
    """
    Given a list of non-intersecting ranges B, ordered in decreasing order, 
    identifies the range that contains the float v.

    If no such bucket is found, raises a Runtime Error.

    Examples:
    value = 0.6
    buckets = [(1, 0.9), (0.9, 0.7), (0.7, 0.6), (0.6, 0.1), (0.1, 0.05), (0.05, 0)]
    -> 
    """
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
    
def matching_prefix_length(s1: str, s2: str, s1_offset: int = 0) -> int:
    """
    Given two strings s1, s2: Finds the number of characters
    in their prefixes which are the same. s1_offset allows us to 
    assign an additional offset to s1 when comparing.

    Examples: 
    s1 = 'quetzalcoatl'
    s2 = 'queueing'
    -> 3

    s1 = 'pleasing'
    s2 = 'leasing'
    -> 0

    s1 = 'pleasing'
    s2 = 'leasing'
    s1_offset = 1
    -> 7

    s1 = '1100101'
    s2 = '1010011'
    -> 1
    """
    output = 0
    while (len(s1) - s1_offset > output and len(s2) > output and s1[output + s1_offset] == s2[output]):
        output += 1
    return output

def invert_digit(digit: str, base: int) -> str:
    """
    Given a digit (single character string) d and base b, returns (d + 1) (mod b).

    Examples:
    digit = '1'
    base = 2
    -> '0'

    digit = '0'
    base = 2
    -> '1'
    """
    return str(
        (int(digit) + 1) % base
    )

def find_prefix(msg: str, prefixes: list[str], msg_offset: int = 0) -> tuple[int, str]:
    """
    Given a long message and a list of prefixes, 
    finds prefix which is a prefix of msg. Returns the index of the prefix and the prefix itself.
    msg_offset allows us to compare msg[msg_offset:] to things
    If none exist, raises a RuntimeError.

    Examples:
    msg = '1010010001001'
    prefixes = ['0', '10000', '10001', '1001', '101000', '101001', '11']
    -> (5, '101001')

    msg = '111000'
    prefixes = ['00', '01', '10', '11']
    msg_offset = 2
    -> (2, '10')
    """
    for ind, prefix in enumerate(prefixes):
        if matching_prefix_length(msg, prefix, msg_offset) == len(prefix):
            return (ind, prefix)
    else:
        raise RuntimeError('Decompression failed! No prefix found!')

def find_longest_match(msg: str, prefixes: list[str], msg_offset: int = 0) -> tuple[tuple[int, str], int]:
    """
    Given a long message and a list of prefixes,
    find prefix with longest match with start of msg. Returns ((prefix index, prefix), length of match).
    If multiple have a tie, raises a RuntimeError.

    msg_offset gives offset to msg so that we effectively compare msg[:msg_offset].

    Examples:
    msg = '10101010'
    prefixes = ['10010', '10100', '00110', '011001']
    msg_offset = 2
    -> ((1, '10100'), 4)
    """

    match_lengths: list[int] = [
        matching_prefix_length(msg, prefix, msg_offset) for prefix in prefixes
    ]

    if len(match_lengths) == 0:
        raise AssertionError('This shouldn\'t be able to happen.')

    max_match = max(match_lengths)

    all_occurances = [
        i for i, el in enumerate(match_lengths) if el == max_match
    ]

    if len(all_occurances) == 0:
        raise AssertionError('This shouldn\'t be able to happen.')
    if len(all_occurances) > 1:
        raise RuntimeError('Prefix can\'t be uniquely decoded!')
    
    prefix_ind = all_occurances[0]
    return ((prefix_ind, prefixes[prefix_ind]), max_match)
    
def expand_bucket_range(error_range_func: any, rng: tuple[float, float], iterations: int = 1):
    if iterations == 0:
        return rng
    (larger, smaller) = rng
    return expand_bucket_range(error_range_func, 
        (error_range_func(larger)[1], error_range_func(smaller)[0]), 
        iterations - 1) 


"""
 ---------- SINGLE TOKEN COMPRESSOR/DECOMPRESSOR ----------
"""

def compress_single_token(
        token: int, pmf: np.ndarray, ref_table: list[str], 
        buckets: list[tuple[float, float]], error_range: any,
        bucket_prefixes: list[str], max_longform: int, base: int) -> str:
    
    """
    Compresses a single token, based on the below encoding scheme. 
    Returns the string compression of the encoding.
    """
    
    bucket_index, target_bucket = identify_bucket(
        pmf[token], buckets
    )


    token_string = ref_table[token]
    expanded_bucket = expand_bucket_range(error_range, target_bucket, 2)
    
    other_strings = [ref_table[i] for i in range(len(pmf)) 
                     if check_range(pmf[i], expanded_bucket) == 0 and i != token
                     ]
        
    minimum_prefixfree_length = max(
        [matching_prefix_length(token_string, s) for s in other_strings], default=0
    ) + 1

    if minimum_prefixfree_length + 1 >= max_longform:
        return bucket_prefixes[bucket_index] + token_string
    else:
        return bucket_prefixes[bucket_index] + token_string[:minimum_prefixfree_length] + invert_digit(
            token_string[minimum_prefixfree_length], base
        )
    


def decompress_single_token(
        msg: str, pmf: np.ndarray, ref_table: list[str], 
        buckets: list[tuple[float, float]], error_range: any,
        bucket_prefixes: list[str], max_longform: int, base: int = 2, msg_offset: int = 0) -> tuple[int, int]:

    """
    Decompresses a single token from a long message. 
    Returns (token number, new message offset).
    """

    (bucket_index, bucket_prefix) = find_prefix(msg, bucket_prefixes, msg_offset)

    msg_offset += len(bucket_prefix)
    expanded_bucket = expand_bucket_range(
        error_range, buckets[bucket_index], 1
    )

    search_space = [
        word for ind, word in enumerate(ref_table) if check_range(pmf[ind], expanded_bucket) == 0
    ]


    ((_, token_string), match_length) = find_longest_match(msg, search_space, msg_offset)

    token = ref_table.index(token_string)

    if match_length == max_longform:
        msg_offset += max_longform
    else:
        msg_offset += 1 + match_length

    return (token, msg_offset)


    
    
    
"""
  ---------- FULL COMPRESSOR/DECOMPRESSOR ----------
"""

def create_package(pmf_generator_1: any, longform_strings: list[str] | dict[int, str], 
                   buckets_cutoffs: list[float] | list[tuple[float, float]], error_range: any,
                   bucket_prefixes: list[str], max_longform: int, tokenizer: any, detokenizer: any, base: int = 2, pmf_generator_2: any = None
                   ) -> tuple[any, any]:
    ref_table = longform_strings
    if isinstance(longform_strings, dict):
        ref_table = [longform_strings[i] for i in range(len(longform_strings))]
    
    buckets = buckets_cutoffs
    if isinstance(buckets[0], int):
        buckets = zip(buckets_cutoffs, buckets_cutoffs[1:])
    
    if pmf_generator_2 is None:
        pmf_generator_2 = pmf_generator_1

    F = error_range

    def compressor(inp: str) -> str:
        all_compressions = []
        tokens = tokenizer(inp)

        prev_tokens = []
        for tok in tokens:
            pmf = pmf_generator_1(prev_tokens)
            all_compressions.append(
                compress_single_token(
                    token=tok, pmf=pmf, ref_table=ref_table, buckets=buckets,
                    error_range=error_range, bucket_prefixes=bucket_prefixes,
                    max_longform=max_longform, base=base
                )
            )
            prev_tokens.append(tok)
        
        return ''.join(all_compressions)

        
    def decompressor(inp: str) -> list[int]:
        msg_offset = 0
        tokens = []
        while msg_offset < len(inp):
            pmf = pmf_generator_2(tokens)
            
            (tok, msg_offset) = decompress_single_token(
                    msg=inp, pmf=pmf, ref_table=ref_table, buckets=buckets,
                    error_range=error_range, bucket_prefixes=bucket_prefixes, 
                    max_longform=max_longform, msg_offset=msg_offset, base=base
                    )
            tokens.append(tok)
        return detokenizer(tokens)
    return (compressor, decompressor)



"""
UNITTEST CASES
"""

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
    
    def test_mpt_01(self):
        self.assertTrue(
            matching_prefix_length(
                '0011010', '0010'
            ) == 3
        )
        self.assertTrue(
            matching_prefix_length(
                '0010', '0010'
            ) == 4
        )
        self.assertTrue(
            matching_prefix_length(
                '001', '0010'
            ) == 3
        )
        self.assertTrue(
            matching_prefix_length(
                '00111', '0011'
            ) == 4
        )
        self.assertTrue(
            matching_prefix_length(
                '001', '1010'
            ) == 0
        )
        self.assertTrue(
            matching_prefix_length(
                'pleasing', 'leasing', 1
            ) == 7
        )

    def test_prefix_finder_01(self):
        self.assertTrue(
            find_prefix('111010011', ['0', '10000', '10001', '1001', '101000', '101001', '11'], 2) == (5, '101001')
        )
    
    def test_inverter_01(self):
        self.assertTrue(
            invert_digit('0', 2) == '1'
        )
        self.assertTrue(
            invert_digit('1', 2) == '0'
        )
        self.assertTrue(
            invert_digit('0', 3) == '1'
        )
        self.assertTrue(
            invert_digit('1', 3) == '2'
        )
        self.assertTrue(
            invert_digit('0', 2) == '1'
        )
        self.assertTrue(
            invert_digit('2', 3) == '0'
        )

    def test_single_token_01(self):
        exp = 0.5
        q = 0.99
        iterations = 100
        num_tokens = 100
        error_func_1 = lambda x: (x*q, x/q)
        power_law_distr_1 = lambda n: (n + 1) ** (exp)
        data1, data2 = generate_synthetic_data(
            num_tokens, iterations, power_law_distr_1, error_func_1
            , 0.1)
        
        tbl = [f"{((i * 71) % 128):7b}".replace(' ', '0') for i in range(num_tokens)]
        buckets = [
            (1, 1/8), (1/8, 1/64), (1/64, 1/512), (1/512, 0)
        ]
        bucket_prefixes = [
            '0', '10', '110', '1110'
        ]
        
        for i in range(iterations):
            pmf1 = data1[i]
            pmf2 = data2[i]
            chosen_token = np.random.choice(np.arange(num_tokens), p=pmf1)
            compression = compress_single_token(
                chosen_token, pmf1, tbl, buckets, error_func_1,
                bucket_prefixes, 7, 2
            )
            decompression, offset = decompress_single_token(
                compression, pmf2, tbl, buckets, error_func_1, 
                bucket_prefixes, 7, 2, 0
                )
            self.assertEqual(chosen_token, decompression)

    def test_package_01(self):
        exp = 1.5
        q = 0.01
        trials = 50
        trial_length = 10
        num_tokens = 128256

        string_lengths = 17
        error_func_1 = lambda x: (x*q, x/q)
        power_law_distr_1 = lambda n: (n + 1) ** (-exp)
        data1, data2 = generate_synthetic_data(
            num_tokens, trials * trial_length, power_law_distr_1, error_func_1
            , 0.1)
        data1 = data1.reshape(trials, trial_length, num_tokens)
        data2 = data2.reshape(trials, trial_length, num_tokens)

        random_prime: int = 97
        
        tbl = [f"{((i * random_prime) % (2 ** string_lengths)):{string_lengths}b}".replace(' ', '0')[::-1] for i in range(num_tokens)]


        norm_constant = 1/sum(power_law_distr_1(i) for i in range(num_tokens))

        print(norm_constant)

        entropy = sum(norm_constant * power_law_distr_1(i) * 
                      math.log2(1 / (power_law_distr_1(i) * norm_constant)) for i in range(num_tokens)
                      )

        print(f'Entropy: {entropy}')

        buckets = []
        bucket_prefixes = [
            '1' * i + '0' for i in range(15)
        ]
        bucket_prefixes[-1] = bucket_prefixes[-1][:-1]

        decay_constant = 1/8
        buckets.append(1)
        for i in range(0, len(bucket_prefixes) - 1):
            buckets.append(
                norm_constant * decay_constant ** (i + 0.5)
                )
        buckets.append(0)
        
        buckets = list(zip(buckets, buckets[1:]))
        
        print(buckets)

        estimated_frequency = [0 for _ in range(len(buckets))]
        estimated_probability = [0 for _ in range(len(buckets))]
        for n in range(num_tokens):
            bucket_num = identify_bucket(power_law_distr_1(n) * norm_constant, buckets=buckets)[0]
            estimated_frequency[bucket_num] += 1
            estimated_probability[bucket_num] += power_law_distr_1(n) * norm_constant
        print(estimated_frequency)
        print(estimated_probability)


        def tokenizer(inpstr: str) -> list[int]:
            output: list[int] = [
                int(float(word)) for word in inpstr.split(' ')
            ]
            return output
        
        def detokenizer(inp: list[int]) -> str:
            return " ".join(
                [
                    str(a) for a in inp
                ]
            )
        
        chosen_tokens = np.zeros((trials, trial_length))
        
        
        for ind, _ in np.ndenumerate(chosen_tokens):
            pmf = data1[ind]
            chosen_tokens[ind] = np.random.choice(np.arange(num_tokens), p=pmf)

        current_trial = 0
            
        def pmf_generator_1(tokens: list[int]) -> np.ndarray:
            return data1[current_trial, len(tokens)]
        
        def pmf_generator_2(tokens: list[int]) -> np.ndarray:
            return data2[current_trial, len(tokens)]
        
        compressor, decompressor = create_package(
            pmf_generator_1=pmf_generator_1, longform_strings=tbl, 
            buckets_cutoffs=buckets, error_range=error_func_1,
            bucket_prefixes=bucket_prefixes, max_longform=string_lengths, tokenizer=tokenizer,
            detokenizer=detokenizer, base=2, pmf_generator_2=pmf_generator_2)
        
        total_length = 0

        for trial in range(trials):
            current_trial = trial
            inp = detokenizer([int(a) for a in
                chosen_tokens[trial]
                ])
            # print(f'Input string: {inp}')
            compression = compressor(inp)
            # print(f'Compression: {compression}')
            total_length += len(compression)
            print(f'Compression Length: {len(compression)}')
            output = decompressor(compression)
            # print(f'Output string: {output}')
            self.assertEqual(inp, output)

        print(f'Performance (in bits): {total_length / (trial_length * trials)}')

        




    


if __name__ == "__main__":
    unittest.main()