import numpy as np
import matplotlib.pyplot as plt
import time 

M = 100      #  bins (m)
T = 30         # Averaging the gap
N_MAX = M * M  # numero massimo di palline 

def calculate_gap(loads, n):
    """Calcola il gap Gn = max(Xi(n) - n/m)."""
    m = len(loads)
    average_load = n / m
    gap = np.max(loads - average_load)
    return max(0, gap) 

#  allocStrategies 

def one_choice_strategy(m, loads=None, *args):
    """Strategia 1: d=1. Scegli un bin a caso."""
    chosen_bin = np.random.randint(0, m)
    return chosen_bin

def two_choice_strategy(m, loads, *args):
    """Strategia 2: d=2. Scegli il bin meno carico."""
    candidates = np.random.choice(m, size=2, replace=False)
    load_1 = loads[candidates[0]]
    load_2 = loads[candidates[1]]
    
    if load_1 < load_2:
        chosen_bin = candidates[0]
    elif load_2 < load_1:
        chosen_bin = candidates[1]
    else: 
        chosen_bin = np.random.choice(candidates)
        
    return chosen_bin

def one_plus_beta_choice_strategy(m, loads, beta):
    """Strategia 3: (1+β)-choice."""
    if np.random.rand() < beta:
        return two_choice_strategy(m, loads)
    else:
        return one_choice_strategy(m) 

def get_load_thresholds(loads):
    """Calcola Mediana (50%), 25% e 75% quartili dei carichi."""
    return np.percentile(loads, [25, 50, 75], method='lower')

def binary_query_strategy(m, loads, k):
    """Strategia 2-choice con k query binarie (Mediana/Quartili)."""
    candidates = np.random.choice(m, size=2, replace=False)
    load_a = loads[candidates[0]]
    load_b = loads[candidates[1]]
    q25, q50, q75 = get_load_thresholds(loads)
    
    is_a_heavy = (load_a >= q50)
    is_b_heavy = (load_b >= q50)
    
    if is_a_heavy != is_b_heavy:
        return candidates[0] if not is_a_heavy else candidates[1]
    
    if k == 1:
        return np.random.choice(candidates)

    if is_a_heavy and is_b_heavy:
        is_a_very_heavy = (load_a >= q75)
        is_b_very_heavy = (load_b >= q75)
        
        if is_a_very_heavy != is_b_very_heavy:
            return candidates[0] if not is_a_very_heavy else candidates[1]
        else:
            return np.random.choice(candidates)
            
    elif not is_a_heavy and not is_b_heavy:
        is_a_heavy_enough = (load_a >= q25)
        is_b_heavy_enough = (load_b >= q25)
        
        if is_a_heavy_enough != is_b_heavy_enough:
            return candidates[0] if not is_a_heavy_enough else candidates[1]
        else:
            return np.random.choice(candidates)
            
    return np.random.choice(candidates)

#simulation

def run_simulation(m, n_max, strategy_func, *strategy_args, is_batched=False, b=1, run_number=1, t_total=1):
    """
    Esegue una singola simulazione con logging di avanzamento.
    """
    loads = np.zeros(m, dtype=int)
    gap_evolution = np.zeros(n_max)
    
    if is_batched:
        outdated_loads = loads.copy()
        
    for n in range(1, n_max + 1):
        
        # allocLogic
        if is_batched:
            ball_in_batch = (n - 1) % b + 1
            if ball_in_batch == 1:
                outdated_loads = loads.copy() 

            if strategy_func.__name__ in ['two_choice_strategy', 'one_plus_beta_choice_strategy']:
                chosen_bin = strategy_func(m, outdated_loads, *strategy_args)
            else:
                chosen_bin = strategy_func(m, loads, *strategy_args)
        
        else: # Standard 
            chosen_bin = strategy_func(m, loads, *strategy_args)

        loads[chosen_bin] += 1
        gap_evolution[n - 1] = calculate_gap(loads, n)
        
    return gap_evolution

def average_multiple_runs(m, n_max, t, strategy_func, label, *strategy_args, is_batched=False, b=1):
    """
    Esegue T simulazioni e calcola la media e deviazione standard del gap, 
    stampando il tempo di esecuzione totale.
    """
    all_gaps = np.zeros((t, n_max))
    start_time_total = time.time()
    
    #print(f"  -> Inizio {label} ({t} run totali)")
    
    for i in range(t):
        start_time_run = time.time()
        
        if is_batched:
            all_gaps[i, :] = run_simulation(m, n_max, strategy_func, *strategy_args, is_batched=True, b=b, run_number=i+1, t_total=t)
        else:
            all_gaps[i, :] = run_simulation(m, n_max, strategy_func, *strategy_args, run_number=i+1, t_total=t)
        
        end_time_run = time.time()
        #print(f"  -> Run {i+1} completato in {end_time_run - start_time_run:.2f} secondi.")

    end_time_total = time.time()
    #print(f"  -> {label} completato. Tempo totale: {end_time_total - start_time_total:.2f} secondi.\n")
    
    avg_gap = np.mean(all_gaps, axis=0)
    std_dev = np.std(all_gaps, axis=0) 
    
    return avg_gap, std_dev

def main():

    global M, T, N_MAX
    
    PLOT_INDICES = np.linspace(0, N_MAX - 1, 300, dtype=int)
    N_VALUES = np.arange(1, N_MAX + 1)[PLOT_INDICES]

    # standard 
    print(f"--- 1. Esperimenti Standard (M={M}, N_MAX={N_MAX}, T={T}) ---")
    STANDARD_RESULTS = {}

    #1CH
    avg_gap_1c, std_gap_1c = average_multiple_runs(M, N_MAX, T, one_choice_strategy, '1-Choice')
    STANDARD_RESULTS['1-Choice'] = (avg_gap_1c, std_gap_1c)

    #2CH 
    avg_gap_2c, std_gap_2c = average_multiple_runs(M, N_MAX, T, two_choice_strategy, '2-Choice')
    STANDARD_RESULTS['2-Choice'] = (avg_gap_2c, std_gap_2c)

    #βCH
    for beta in [0.2, 0.5, 0.8]:
        label = f'(1+{beta})-Choice'
        avg, std = average_multiple_runs(M, N_MAX, T, one_plus_beta_choice_strategy, label, beta)
        STANDARD_RESULTS[label] = (avg, std)

    #Plot1 
    plt.figure(figsize=(12, 7))
    for label, (avg_gap, std_dev) in STANDARD_RESULTS.items():
        plt.plot(N_VALUES, avg_gap[PLOT_INDICES], label=label)
        if T > 1:
            plt.fill_between(N_VALUES, 
                             (avg_gap - std_dev)[PLOT_INDICES], 
                             (avg_gap + std_dev)[PLOT_INDICES], 
                             alpha=0.15)

    plt.axvline(M, color='k', linestyle='--', linewidth=1, label=f'Light-Load ($n=m$)')
    plt.axvline(N_MAX, color='r', linestyle=':', linewidth=1)
    plt.title(f'Gap Evolution in Standard Setting (T={T})')
    plt.xlabel('number of balls $n$')
    plt.ylabel(f'Gap $\\bar{{G}}_n$')
    plt.xscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig('Plots/grafico_standard.png') 
    plt.close()


    #b-batched
    print(f"--- 2. Esperimenti b-Batched (T={T}) ---")
    BATCHED_RESULTS = {}
    B_VALUES = [M, 10 * M] 

    for b in B_VALUES:
        if b > N_MAX: continue
        
        label = f'2-Choice (b={b}/m)'
        avg, std = average_multiple_runs(M, N_MAX, T, two_choice_strategy, label, is_batched=True, b=b)
        BATCHED_RESULTS[label] = (avg, std)

    #Plot2
    plt.figure(figsize=(12, 7))

    avg_gap_2c_std, std_gap_2c_std = STANDARD_RESULTS['2-Choice']
    plt.plot(N_VALUES, avg_gap_2c_std[PLOT_INDICES], label='2-Choice (Standard, b=1)', linestyle='-', color='blue')
    if T > 1:
        plt.fill_between(N_VALUES, 
                         (avg_gap_2c_std - std_gap_2c_std)[PLOT_INDICES], 
                         (avg_gap_2c_std + std_gap_2c_std)[PLOT_INDICES], 
                         alpha=0.15, color='blue')

    for label, (avg_gap, std_dev) in BATCHED_RESULTS.items():
        plt.plot(N_VALUES, avg_gap[PLOT_INDICES], label=label, linestyle='--')
        if T > 1:
            plt.fill_between(N_VALUES, 
                             (avg_gap - std_dev)[PLOT_INDICES], 
                             (avg_gap + std_dev)[PLOT_INDICES], 
                             alpha=0.1)

    plt.axvline(M, color='k', linestyle='--', linewidth=1, label=f'Light-Load ($n=m$)')
    plt.axvline(N_MAX, color='r', linestyle=':', linewidth=1)
    plt.title(f' b-Batched (T={T})')
    plt.xlabel('Number of Balls $n$ (Scala Logaritmica)')
    plt.ylabel(f'Gap $\\bar{{G}}_n$')
    plt.xscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig('Plots/grafico_batched.png') 
    plt.close()


    #BinQueries
    print(f"--- 3. Esperimenti con Query Binarie (T={T}) ---")
    QUERY_RESULTS = {}

    # k=1 Query
    avg_gap_q1, std_gap_q1 = average_multiple_runs(M, N_MAX, T, binary_query_strategy, '2-Choice (k=1 Query)', 1)
    QUERY_RESULTS['2-Choice (k=1 Query)'] = (avg_gap_q1, std_gap_q1)

    # k=2 Query
    avg_gap_q2, std_gap_q2 = average_multiple_runs(M, N_MAX, T, binary_query_strategy, '2-Choice (k=2 Query)', 2)
    QUERY_RESULTS['2-Choice (k=2 Query)'] = (avg_gap_q2, std_gap_q2)

    #Plot3
    plt.figure(figsize=(12, 7))

    plt.plot(N_VALUES, avg_gap_2c_std[PLOT_INDICES], label='2-Choice (Standard, k=Inf Query)', linestyle='-', color='blue')
    if T > 1:
        plt.fill_between(N_VALUES, 
                         (avg_gap_2c_std - std_gap_2c_std)[PLOT_INDICES], 
                         (avg_gap_2c_std + std_gap_2c_std)[PLOT_INDICES], 
                         alpha=0.15, color='blue')

    for label, (avg_gap, std_dev) in QUERY_RESULTS.items():
        plt.plot(N_VALUES, avg_gap[PLOT_INDICES], label=label, linestyle='--')
        if T > 1:
            plt.fill_between(N_VALUES, 
                             (avg_gap - std_dev)[PLOT_INDICES], 
                             (avg_gap + std_dev)[PLOT_INDICES], 
                             alpha=0.1)

    plt.axvline(M, color='k', linestyle='--', linewidth=1, label=f'Light-Load ($n=m$)')
    plt.axvline(N_MAX, color='r', linestyle=':', linewidth=1)
    plt.title(f'Binary Query (T={T})')
    plt.xlabel('Number of balls $n$')
    plt.ylabel(f'Gap $\\bar{{G}}_n$')
    plt.xscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig('Plots/grafico_query.png') 
    plt.close()

    #finalSumup
    print("\n--- Sumup ---")
    final_results = {}
    final_results.update(STANDARD_RESULTS)
    final_results.update(BATCHED_RESULTS)
    final_results.update(QUERY_RESULTS)

    print("{:<30} {:>10}".format("Strategy", f"Medium Gap (n={N_MAX})"))
    print("-" * 42)
    for label in sorted(final_results.keys()):
        gap_at_n_max = final_results[label][0][N_MAX - 1]
        print("{:<30} {:>10.2f}".format(label, gap_at_n_max))

if __name__ == "__main__":
    main()
