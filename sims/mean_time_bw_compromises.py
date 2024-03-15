import multiprocessing as mp
import time
from typing import List

import matplotlib
import pandas as pd
import simpy
from cycler import cycler
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from sims import primitives
from sims.primitives import SIM_TIME


class Simulation(object):
    def __init__(self, ttl_percents: List[List[int]], figure: plt.Figure) -> None:
        super().__init__()
        self.ttl_percents = ttl_percents
        self.result_dict = {}
        self.figure = figure

    @staticmethod
    def compute_ttl(percent_sim_time: int) -> int:
        return round(SIM_TIME * (percent_sim_time / 100.0))

    @property
    def number_of_figures(self):
        count = len(self.ttl_percents)
        return count * 100 + 11

    def run(self):
        with mp.Pool(mp.cpu_count()) as pool:
            self._run(pool)
            pool.close()
            pool.join()

    def _run(self, pool: mp.Pool):
        for index, ttl_percents in enumerate(self.ttl_percents):
            for ttl_percent in ttl_percents:
                for i in range(20):
                    current_ttl = self.compute_ttl(ttl_percent)
                    if current_ttl not in self.result_dict:
                        self.result_dict[current_ttl] = []
                    pool.apply_async(self._simulation_code, args=(current_ttl,),
                                     callback=lambda r: self.result_dict[r[0]].append(r[1]))

    def draw(self):
        for index, ttl_percents in enumerate(self.ttl_percents):
            ax: Axes = self.figure.add_subplot(self.number_of_figures + index)
            for ttl_percent in ttl_percents:
                current_ttl: int = self.compute_ttl(ttl_percent)
                series: list = self.result_dict[current_ttl]
                concated_series: pd.Series = pd.concat(series, axis=1)

                mean_columns = concated_series.mean(axis=1)

                final_rolling_mean: pd.DataFrame = mean_columns.rolling(100).mean()
                ax.plot(final_rolling_mean, label="TTL {}% of Sim time".format(ttl_percent), linewidth=3)
            plt.legend()

    def _simulation_code(self, current_ttl):
        env = simpy.Environment()
        cloud_system = CloudSystem(env)
        cloud_system.current_ttl = current_ttl
        attacker = DumbAttacker(env, cloud_system.in_service_cfs)
        env.process(cloud_system.run())
        env.process(attacker.run())
        env.run(until=SIM_TIME)
        print("Sim TTL {} complete".format(current_ttl))
        result = attacker.attack_times.diff()
        return current_ttl, result


class DumbAttacker(primitives.DumbAttacker):
    def __init__(self, env: simpy.Environment, cfs_in_service: simpy.Store) -> None:
        super().__init__(env, cfs_in_service)
        self.attack_times = pd.Series()

    def _attack_succeded(self, cfs: primitives.CFS):
        super()._attack_succeded(cfs)
        self.attack_times = self.attack_times.append(pd.DataFrame(data=[self.env.now], index=[self.env.now]))


class CloudSystem(primitives.CloudSystem):
    pass


def complete_simulation():
    figure = plt.figure()
    figure.set_size_inches((15, 10))

    sim = Simulation(ttl_percents=[
        [100, 50, 25, 10, 8, 3]
    ], figure=figure)
    sim.run()
    sim.draw()


def partial_test_simulation():
    pass


if __name__ == '__main__':
    start_time = time.time()
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 22}

    matplotlib.rc('font', **font)
    default_cycler = (cycler(color=['r', 'g', 'b', 'y']) +
                      cycler(linestyle=['-', '--', ':', '-.']))
    plt.rc('axes', prop_cycle=default_cycler)
    complete_simulation()
    print("Execution took: {} seconds".format(round(time.time() - start_time)))
    plt.savefig('mean_time_bw_compromises.png', bbox_inches='tight')

    plt.show()
