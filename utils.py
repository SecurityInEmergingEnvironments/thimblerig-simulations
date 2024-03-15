from scipy.special import binom as binom_coeff


def probability_of_event_occuring_atleast_k_times(k: float, trials: int, probability_of_occuring_once: float):
    p = probability_of_occuring_once
    n = trials
    binomial_coefficient = binom_coeff(n, k)
    result = binomial_coefficient * (p ** k) * ((1 - p) ** (n - k))
    return result


def probability_of_event_occuring_k_or_more_times(K: int, trials: int, probability_of_occuring_once: float):
    return sum((probability_of_event_occuring_atleast_k_times(k, trials, probability_of_occuring_once)
                for k in range(K, trials + 1)))
