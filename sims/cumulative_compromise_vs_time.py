import multiprocessing as mp
import time

import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler

from sims.primitives import *
from sims.primitives import SIM_TIME

ATTACKER_MIN_USABLE_TTL = 10

# Number of times a simulation is run to aggregate results
NUMBER_OF_ITERATIONS = 20

ROLLING_MEAN_SIZE = 10


def simulate(current_ttl):
    env = simpy.Environment()
    cloud_system = CloudSystem(env)
    cloud_system.current_ttl = current_ttl
    attacker = DumbAttacker(env, cloud_system.in_service_cfs)
    attacker.attacker_min_usable_time = 0.005 * SIM_TIME
    env.process(cloud_system.run())
    env.process(attacker.run())
    env.run(until=SIM_TIME)
    print("Sim TTL {} complete".format(current_ttl))
    return current_ttl, attacker.cumulative_usable_time


if __name__ == '__main__':
    start_time = time.time()
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 22}

    matplotlib.rc('font', **font)
    default_cycler = (cycler(color=['r', 'g', 'b', 'y']) +
                      cycler(linestyle=['-', '--', ':', '-.']))
    plt.rc('axes', prop_cycle=default_cycler)
    for index, ttl_percents in enumerate([
        [100, 80, 60, 40],
        [20, 10, 5, 2],
        [1, 0.5, 0.3]
    ]):
        data_dict = {}
        with mp.Pool(mp.cpu_count()) as pool:
            for i in range(20):
                for percent_sim_time in ttl_percents:
                    current_ttl = round(SIM_TIME * (percent_sim_time / 100.0))
                    if current_ttl not in data_dict:
                        data_dict[current_ttl] = []
                    pool.apply_async(simulate,
                                     args=(current_ttl,),
                                     callback=lambda r: data_dict[r[0]].append(r[1]),
                                     error_callback=lambda e: print(e))

            pool.close()
            pool.join()

        fig = plt.figure()
        fig.set_size_inches((15, 10))
        for ttl, series in data_dict.items():
            # Axis 1 means combine as columns
            concated_series: pd.Series = pd.concat(series, axis=1)
            mean_columns = concated_series.mean(axis=1)
            # mean_columns.to_csv('images/ttl_{}.csv'.format(ttl))
            plt.plot(mean_columns.cumsum(), label="TTL %d" % ttl, linewidth=3)
        plt.xlabel("SIM Time")
        plt.ylabel("Cumulative exploitable time")
        plt.title("TTLs {}%".format(ttl_percents))
        plt.legend()
        plt.savefig('images/cum_sum_of_exploitable_time_ttls{}.png'.format(ttl_percents),
                    bbox_inches='tight')
    # plt.show()
    print("Execution took: {}".format(round(time.time() - start_time)))
