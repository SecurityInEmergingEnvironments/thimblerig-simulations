import multiprocessing as mp
import numbers
import os
import time
from multiprocessing.pool import ApplyResult
from typing import List, Tuple

import numpy as np
import pandas as pd
import simpy
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from sims import primitives
from sims.primitives import SIM_TIME

RECORDS_PER_UNIT_TIME = 5


def compute_impact(records: numbers.Number) -> numbers.Number:
    # return 107 * records
    log_impact = 7.68 + 0.76 * np.log(records)
    impact = round(np.power(np.e, log_impact))
    return impact


class DumbAttacker(primitives.DumbAttacker):
    # def _was_attack_successful(self, cfs: CFS):
    #     return not (cfs.ttl_expired or cfs.is_compromised)
    pass


class CloudSystem(primitives.CloudSystem):
    pass


class SingleSimulation(object):
    def __init__(self, ttl: int,
                 plot_axes: Axes = plt,
                 iteration_count: int = 20,
                 pool: mp.Pool = None,
                 sim_time: int = SIM_TIME) -> None:
        self.ttl = ttl
        self.plot_axes = plot_axes
        self.iteration_count = iteration_count
        self.pool = pool
        self.sim_time = sim_time
        self.current_results = None
        self.label = "TTL {}".format(ttl)

    def draw(self):
        attacker_usable_times: List[pd.Series] = []
        for result in self.current_results:
            ttl, usable_time = result.get()
            attacker_usable_times.append(usable_time)
        combined_usable_series: pd.Series = pd.concat(attacker_usable_times, axis=1)
        column_mean = combined_usable_series.mean(axis=1)
        self.plot_axes.plot(column_mean.cumsum(), label=self.label)

    def _simulate(self) -> Tuple[int, pd.Series]:
        env = simpy.Environment()
        cloud_system = CloudSystem(env)
        cloud_system.current_ttl = self.ttl
        attacker = DumbAttacker(env, cloud_system.in_service_cfs)
        env.process(cloud_system.run())
        env.process(attacker.run())
        env.run(until=self.sim_time)
        print("Sim TTL {} complete".format(self.ttl))
        usable_time = attacker.cumulative_usable_time
        return self.ttl, usable_time

    def _run(self, pool) -> List[ApplyResult]:
        result_futures = []
        for i in range(self.iteration_count):
            result_future = pool.apply_async(self._simulate)
            result_futures.append(result_future)
        return result_futures

    def run(self):
        if self.pool:
            result_futures = self._run(self.pool)
        else:
            with mp.Pool(mp.cpu_count()) as pool:
                result_futures = self._run(pool)
                pool.close()
                pool.join()
        self.current_results = result_futures
        return result_futures


class Simulation(object):
    def __init__(self, ttl_percents: List[List[Tuple[int, str]]], figure: plt.Figure) -> None:
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
            for ttl_percent, linestyle in ttl_percents:
                for i in range(20):
                    current_ttl = self.compute_ttl(ttl_percent)
                    if current_ttl not in self.result_dict:
                        self.result_dict[current_ttl] = []
                    pool.apply_async(self._simulation_code, args=(current_ttl,),
                                     callback=lambda r: self.result_dict[r[0]].append(r[1]))

    def draw(self):
        for index, ttl_percents in enumerate(self.ttl_percents):
            ax: Axes = self.figure.add_subplot(self.number_of_figures + index)
            for ttl_percent, linestyle in ttl_percents:
                current_ttl: int = self.compute_ttl(ttl_percent)
                series: list = self.result_dict[current_ttl]
                concated_series: pd.Series = pd.concat(series, axis=1)

                mean_columns: pd.Series = concated_series.mean(axis=1)
                impact_series = mean_columns.cumsum().map(
                        lambda usable_time: compute_impact(usable_time * RECORDS_PER_UNIT_TIME))
                ax.plot(impact_series, label="TTL {}% of Sim time".format(ttl_percent), linestyle=linestyle, linewidth=3)
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
        result = attacker.cumulative_usable_time
        return current_ttl, result


def complete_simulation():
    figure = plt.figure()
    figure.set_size_inches((15, 10))

    sim = Simulation(ttl_percents=[
        [(100, ':'), (50, '-.'), (25, '--'), (6, '-')],

    ], figure=figure)
    sim.run()
    sim.draw()


def partial_test_simulation():
    figure = plt.figure()
    ttl = round(SIM_TIME * 0.5)
    simulation = SingleSimulation(ttl)
    results = simulation.run()
    simulation.draw()
    plt.show()


if __name__ == '__main__':
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 22}

    matplotlib.rc('font', **font)


    start_time = time.time()
    complete_simulation()
    print("Execution took: {} seconds".format(round(time.time() - start_time)))
    plt.xlabel("Simulation time")
    plt.ylabel("Cost in dollars")
    plt.title("Cost comparison of getting attacked at various TTLs")
    plt.yscale('log')
    # plt.figtext(x=0.5, y=0.5, s="Records per unit time = {}".format(RECORDS_PER_UNIT_TIME))
    plt.savefig(os.path.basename(__file__).replace('.py', '.png'), bbox_inches='tight')
    plt.show()
