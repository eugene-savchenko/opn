import pandas as pd
import numpy as np
import sympy
import itertools
import os
import multiprocessing
import argparse

# --- Проверка формы N = p^k * m^2 ---
def check_perfect_odd_form(n, factorization):
    odd_exponent_indices = [i for i, (p, k) in enumerate(factorization.items()) if k % 2 == 1]
    if len(odd_exponent_indices) != 1:
        return False

    p, k = list(factorization.items())[odd_exponent_indices[0]]

    if p % 4 != 1 or k % 4 != 1:
        return False

    if n % 105 == 0:
        return False

    if not (n % 12 == 1 or n % 468 == 117 or n % 324 == 81):
        return False

    primes_sorted = sorted(factorization.keys(), reverse=True)

    if primes_sorted:
        if primes_sorted[0] >= (3 * n) ** (1 / 3):
            return False
    if len(primes_sorted) >= 2:
        if primes_sorted[1] >= (2 * n) ** (1 / 5):
            return False
    if len(primes_sorted) >= 3:
        if primes_sorted[2] >= (2 * n) ** (1 / 6):
            return False

    return True

# Получение делителей из факторизации
def get_divisors_from_factorization(factorization):
    primes = list(factorization.keys())
    exponents = list(factorization.values())
    
    ranges = [range(k + 1) for k in exponents]

    divisors = []
    for powers in itertools.product(*ranges):
        divisor = 1
        for p, power in zip(primes, powers):
            divisor *= p ** power
        divisors.append(divisor)
    return sorted(divisors)[:-1]

# Обработка одного числа
def process_number(n):
    factorization_dict = sympy.factorint(n)
    factorization = list(factorization_dict.items())

    divisors = get_divisors_from_factorization(factorization_dict)

    sum_divs = sum(divisors)
    max_div = max(divisors) if divisors else None
    count_div = len(divisors)
    median_div = np.median(divisors) if divisors else None
    mean_div = np.mean(divisors) if divisors else None

    prime_factors_repeated = []
    for p, k in factorization:
        prime_factors_repeated.extend([p] * k)

    prime_factors_unique = list(factorization_dict.keys())

    count_prime_factors_repeated = len(prime_factors_repeated)
    median_prime_repeated = np.median(prime_factors_repeated) if prime_factors_repeated else None
    mean_prime_repeated = np.mean(prime_factors_repeated) if prime_factors_repeated else None
    max_prime = max(prime_factors_unique) if prime_factors_unique else None
    min_prime = min(prime_factors_unique) if prime_factors_unique else None

    degrees = list(factorization_dict.values())
    max_degree = max(degrees) if degrees else None
    min_degree = min(degrees) if degrees else None

    has_odd_total_prime_factors = count_prime_factors_repeated % 2 == 1
    satisfies_perfect_odd_form = check_perfect_odd_form(n, factorization_dict)

    abs_rel_sum = abs(sum_divs / n - 1)

    if abs_rel_sum < 0.05 and satisfies_perfect_odd_form:
        return {
            'number': n,
            'sum_divisors': sum_divs,
            'max_divisor': max_div,
            'count_divisors': count_div,
            'median_divisor': median_div,
            'mean_divisor': mean_div,
            'count_prime_factors_repeated': count_prime_factors_repeated,
            'count_prime_factors_unique': len(prime_factors_unique),
            'max_prime_factor': max_prime,
            'min_prime_factor': min_prime,
            'median_prime_factors_repeated': median_prime_repeated,
            'mean_prime_factors_repeated': mean_prime_repeated,
            'relative_sum_divisors': sum_divs / n - 1,
            'abs_relative_sum_divisors': abs_rel_sum,
            'relative_max_divisor': max_div / n if max_div else None,
            'relative_median_divisor': median_div / n if median_div else None,
            'relative_mean_divisor': mean_div / n if mean_div else None,
            'relative_median_prime_factors_repeated': median_prime_repeated / n if median_prime_repeated else None,
            'relative_mean_prime_factors_repeated': mean_prime_repeated / n if mean_prime_repeated else None,
            'relative_max_prime_factor': max_prime / n if max_prime else None,
            'relative_min_prime_factor': min_prime / n if min_prime else None,
            'max_min_prime_ratio': max_prime / min_prime if max_prime and min_prime else None,
            'factorization': factorization,
            'max_prime_degree': max_degree,
            'min_prime_degree': min_degree,
            'has_odd_total_prime_factors': has_odd_total_prime_factors,
            'satisfies_perfect_odd_form': satisfies_perfect_odd_form
        }
    else:
        return None

# Обработка блока чисел и сохранение
def process_block(start, end, block_id, save_dir="results"):
    filename = f"{save_dir}/block_{block_id}.csv"
    if os.path.exists(filename):
        print(f"Блок {block_id} уже обработан. Пропускаем.")
        return

    print(f"Обработка блока {block_id}: {start} - {end}")
    
    with multiprocessing.Pool() as pool:
        results = pool.map(process_number, range(start, end, 2))

    filtered_results = [res for res in results if res]

    df = pd.DataFrame(filtered_results)
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"Сохранён блок {block_id} -> {filename}")

# Объединение всех блоков
def combine_blocks(save_dir="results", final_filename="final_results.csv"):
    all_blocks = []
    for file in sorted(os.listdir(save_dir)):
        if file.endswith(".csv"):
            df_block = pd.read_csv(os.path.join(save_dir, file))
            all_blocks.append(df_block)

    df_full = pd.concat(all_blocks, ignore_index=True)
    df_full.sort_values(by='abs_relative_sum_divisors', inplace=True)
    df_full.to_csv(final_filename, index=False)
    print(f"Финальный файл сохранён: {final_filename}")

# Управляющая функция
def main():
    parser = argparse.ArgumentParser(description="Process number ranges.")
    parser.add_argument('--start', type=int, default=3, help='Start of range')
    parser.add_argument('--end', type=int, default=1000001, help='End of range')
    parser.add_argument('--block_size', type=int, default=1000000, help='Block size')
    args = parser.parse_args()

    print(f"Starting calculation from {args.start} to {args.end} with block size {args.block_size}...")

    blocks = [(i, min(i + args.block_size, args.end)) for i in range(args.start, args.end, args.block_size)]

    for _, (block_start, block_end) in enumerate(blocks):
        block_id = block_start // args.block_size
        if not block_start % 2:
            block_start += 1
        process_block(block_start, block_end, block_id)

    combine_blocks()

if __name__ == "__main__":
    main()
