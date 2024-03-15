# Goals adaptive defender based on attacker's rate.
# TODO Complete knowledge
# TODO Defender knows about attacker's behaviour
# TODO Attacker attacks at Nash Equilibrium rate
# TODO Successful attacks are marked on a series
# TODO A Process each for attacker and defender
import multiprocessing as mp
import numbers
import random
import secrets
import time
import traceback
from collections import deque
from typing import List, Callable

import numpy as np
import pandas as pd
import simpy
from matplotlib import pyplot as plt
from pandas.computation.ops import UndefinedVariableError
from simpy import Environment


def _decision(probability: float):
    assert 0.0 < probability <= 1
    return secrets.randbelow(101) < (probability * 100)


CFS_COUNT = 100
BENEFIT_USAGE = 0.6
COST_VICTIM_PER_UNIT_TIME = 2
COST_RESET = 0.047

NASH_EQUILIBRIUM_ATTACK_PROBABILITY = 0.09976991338

INITIAL_TTL = 50
LENGTH_OF_ATTACK = 24

SIM_TIME = 434 * 24
ACCEPTABLE_UTILITY_RANGE = (2, 20)
EVENT_NAME_SUCCESSFUL_ATTACK = 'ATTACK_SUCCESSFUL'
EVENT_NAME_FAILED_ATTACK = 'ATTACK_FAILED'
MIN_TTL = 20
SCAN_WINDOW = 1000


def compute_impact(records: numbers.Number) -> numbers.Number:
    # return 107 * records
    log_impact = 7.68 + 0.76 * np.log(records)
    impact = round(np.power(np.e, log_impact))
    return impact


class CFS(object):
    def __init__(self, env: Environment, ttl: int):
        self.env = env
        self.ttl = ttl
        assert ttl > 0
        self.started_at = env.now
        self.is_compromised = False
        self.compromised_at = None
        self.exploitable_time = 0

    def mark_compromised(self):
        self.is_compromised = True
        self.compromised_at = self.env.now
        self.exploitable_time = self.time_left

    @property
    def shutdown_at(self) -> int:
        return self.started_at + self.ttl

    @property
    def time_left(self) -> int:
        return self.shutdown_at - self.env.now

    @property
    def is_running(self) -> bool:
        return self.env.now >= self.started_at

    @property
    def is_shutdown(self) -> bool:
        return self.env.now - self.started_at >= self.ttl

    @property
    def as_dict(self):
        return {
            'ttl':              self.ttl,
            'started_at':       self.started_at,
            'compromised_at':   self.compromised_at,
            'is_compromised':   self.is_compromised,
            'exploitable_time': self.time_left,
            '_generated_at_':   self.env.now
        }


class NashEquilibriumAttacker(object):
    def __init__(self, env: Environment, cfs_store: simpy.Store, event_calbacks: List[Callable] = None) -> None:
        super().__init__()
        self.env = env
        self.attack_probability = NASH_EQUILIBRIUM_ATTACK_PROBABILITY
        self.cfs_store = cfs_store
        self.length_of_an_attack = LENGTH_OF_ATTACK
        self.event_callbacks = event_calbacks if event_calbacks else []

    @staticmethod
    def was_attack_successful(cfs: CFS):
        return cfs.is_running and (not (cfs.is_shutdown))

    def attack_succeeded(self, cfs: CFS):
        cfs.mark_compromised()
        for callback in self.event_callbacks:
            callback(EVENT_NAME_SUCCESSFUL_ATTACK, cfs.as_dict)

    def attack_failed(self, cfs: CFS):
        for callback in self.event_callbacks:
            callback(EVENT_NAME_FAILED_ATTACK, cfs.as_dict)

    @property
    def random_cfs(self):
        if _decision(self.attack_probability) and len(self.cfs_store.items) > 0:
            return random.choice(self.cfs_store.items)
        else:
            return None

    def run(self):
        while True:
            yield self.env.timeout(1)
            cfs_of_choice = self.random_cfs
            if cfs_of_choice:
                yield self.env.timeout(self.length_of_an_attack)
                if self.was_attack_successful(cfs_of_choice):
                    self.attack_succeeded(cfs_of_choice)
                else:
                    self.attack_failed(cfs_of_choice)


class Defender(object):
    def __init__(self, env: Environment) -> None:
        super().__init__()
        self.env = env
        self.current_ttl = INITIAL_TTL
        self.cfs_count = CFS_COUNT
        self.scan_window = SCAN_WINDOW
        self.cfs_store = simpy.Store(env, capacity=self.cfs_count)
        self.initialize_cfs_store()
        self.compromise_data = pd.DataFrame()
        self.utility_series = {}
        self.ttl_series = {
            self.env.now: self.current_ttl
        }

        self.is_adaptive = False

    def update_ttl(self, ttl):
        if ttl < MIN_TTL:
            self.current_ttl = MIN_TTL
        else:
            self.current_ttl = ttl
        self.ttl_series[self.env.now] = self.current_ttl

    def build_cfses(self, count: int, min_start_variance: int = 0, max_start_variance: int = 0) -> List[CFS]:
        cfses = []
        for i in range(count):
            cfs = CFS(self.env, self.current_ttl)
            cfs.started_at += random.randint(min_start_variance, max_start_variance)
            cfses.append(cfs)
        return cfses

    def initialize_cfs_store(self):
        deque(map(self.cfs_store.put, self.build_cfses(self.cfs_count, 0, 10)))

    def run(self):
        while True:
            yield self.env.timeout(1)
            self.load_balancer()
            if self.env.now >= SCAN_WINDOW and self.env.now % 10 == 0:
                self.adapt()

    def load_balancer(self):
        for cfs in self.cfs_store.items:
            if cfs.is_shutdown:
                self.cfs_store.items.remove(cfs)
        count_of_inservice_cfs = len(self.cfs_store.items)
        if count_of_inservice_cfs < self.cfs_count:
            count_cfs_needed = self.cfs_count - count_of_inservice_cfs
            deque(map(self.cfs_store.put, self.build_cfses(count_cfs_needed, 0, 2)))

    def mean_exploitable_time(self, time_units: int):
        data_points = self.attacks_in_last(time_units)
        records = data_points.to_dict('records')
        exploitable_time = pd.Series(map(lambda x: x['exploitable_time'], records))
        return exploitable_time.mean()

    def data_points_in_last(self, time_units: int):
        start = self.env.now - time_units
        return self.compromise_data.query('{} <= _generated_at_ <= {}'.format(start, self.env.now))

    def attacks_in_last(self, time_units: int):
        data_points = self.data_points_in_last(time_units)
        return data_points.query('is_compromised==True')

    @property
    def expected_attack_probability(self):
        exploitable_time = self.mean_exploitable_time(self.scan_window)
        step1 = COST_RESET + (BENEFIT_USAGE * SIM_TIME) - (BENEFIT_USAGE * self.current_ttl)
        step2 = (COST_VICTIM_PER_UNIT_TIME * SIM_TIME) - (COST_VICTIM_PER_UNIT_TIME * exploitable_time)
        return step1 / step2

    @property
    def actual_attack_probability(self):
        # TODO why isn't attack probability converging to NASH_EQUILIBRIUM_ATTACK_PROBABILITY
        num_attacks = len(self.attacks_in_last(self.scan_window))
        return num_attacks / self.scan_window

    def adapt(self):
        # utility_change = self.utility_change
        expected_attack_probability = self.expected_attack_probability
        actual_attack_probability = self.actual_attack_probability

        difference = NASH_EQUILIBRIUM_ATTACK_PROBABILITY - actual_attack_probability
        threshold = 0
        if difference > threshold:
            new_ttl = self.current_ttl + 1
        elif difference < threshold:
            new_ttl = self.current_ttl - 1
        else:
            new_ttl = self.current_ttl

        self.update_ttl(new_ttl)

    def record_utility(self):
        self.utility_series[self.env.now] = self.current_utility

    @property
    def utility_change(self):
        return pd.Series(self.utility_series)[-self.scan_window:].diff().dropna().mean()

    @property
    def current_utility(self):
        """
            Costs
                - C_Vic * ExpT
                - C_Reset
            Benefits
                + B_Use * TTL
        :return: 
        """
        start = self.env.now - self.scan_window
        start = start if start > 0 else 0
        try:
            query = self.compromise_data.query('{} <= _generated_at_ <= {}'.format(start, self.env.now))

            current_utility = []
            for data_dict in query.to_dict('records'):
                utility = 0
                if data_dict['is_compromised']:
                    # Victim Cost
                    utility -= data_dict['exploitable_time'] * COST_VICTIM_PER_UNIT_TIME
                # ResetCost
                utility -= COST_RESET
                # Usage benefit
                utility += data_dict['ttl'] * BENEFIT_USAGE
                current_utility.append(utility)
            return pd.Series(current_utility).mean()
        except UndefinedVariableError:
            return np.NaN

    def attack_event_callback(self, event_name: str, data_dict: dict):
        self.compromise_data = (
            self.compromise_data.append(
                    pd.DataFrame([data_dict])))
        self.record_utility()


def process_results():
    pass


def simulate_ttl(ttl):
    env = simpy.Environment()
    defender = Defender(env)
    defender.current_ttl = ttl
    attacker = NashEquilibriumAttacker(env=env, cfs_store=defender.cfs_store,
                                       event_calbacks=[defender.attack_event_callback])

    env.process(defender.run())
    env.process(attacker.run())
    env.run(until=SIM_TIME)
    print("SIM TTL {}".format(ttl))
    return ttl, pd.Series(defender.utility_series), pd.Series(defender.ttl_series)


def error_callback(e):
    traceback.print_exc()


def simulate():
    results = []
    ttl_ranges = [1000, 5000]
    with mp.Pool(mp.cpu_count()) as pool:
        for ttl in ttl_ranges:
            pool.apply_async(
                    simulate_ttl,
                    (ttl,),
                    callback=lambda result: results.append(result), error_callback=lambda e: print(e))
        pool.close()
        pool.join()

    fig = plt.figure()
    fig.set_size_inches((15, 10))
    utility_ax = fig.add_subplot(211)
    plt.xlabel('SIM Time')
    plt.ylabel('Utility')
    ttl_ax = fig.add_subplot(212)
    plt.xlabel('SIM Time')
    plt.ylabel('TTL')
    for ttl, series1, series2 in results:
        utility_ax.plot(series1, '--', label='U for initial ttl {}'.format(ttl))
        ttl_ax.plot(series2, '-', label='TTL with initial {}'.format(ttl))

    utility_ax.legend()
    # utility_ax.xlabel('Simulation time')
    # utility_ax.ylabel('Utility')

    ttl_ax.legend()
    # ttl_ax.xlabel('Simulation time')
    # ttl_ax.ylabel('TTL')

    plt.savefig('images/nash_equilibrium_initial_ttl_{}.png'.format(ttl_ranges), bbox_inches='tight')


if __name__ == '__main__':
    start_time = time.time()
    simulate()
    print("Execution took: {} seconds".format(round(time.time() - start_time)))
    # plt.legend()
    plt.show()



# Victim cost
# - Number of Attacks
# - Attack probability
# - TTL
# Reset Cost
# - TTL
# - Request Rate [Assume constant]
