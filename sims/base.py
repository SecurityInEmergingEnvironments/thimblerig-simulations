import csv
import logging
import multiprocessing
import random
import secrets
from collections import Iterable, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import simpy

MAX_POOL_SIZE = 10
MAX_CFS_IN_SERVICE = 10
MIN_TTL = 10
ATTACKER_MIN_USABLE_TTL = 5

INITIAL_SCAN_PROBABILITY = 0.3
INITIAL_ATTACK_PROBABILITY = 0.009
INITIAL_TTL = 100

# Mean tolerable attack rate
ATTACK_THRESHOLD = 15

# Adaptive Algorithm
IS_ADAPTIVE = True
SIM_TIME = 10000

# MIN_TTL = 5
#
# INITIAL_SCAN_PROBABILITY = 0.3
# INITIAL_ATTACK_PROBABILITY = 0.2
# INITIAL_TTL = 100
#
# # Mean tolerable attack rate
# ATTACK_THRESHOLD = 15

SIM_TIME = 434
log = logging.getLogger()
log.setLevel(logging.DEBUG)


def _decision(probability):
    return secrets.randbelow(101) < (probability * 100)


class CFS(object):
    def __init__(self, env) -> None:
        super().__init__()
        self.env = env
        self.ttl = None
        self.started_at = None

    @property
    def shutdown_at(self):
        if self.started_at and self.ttl:
            return self.started_at + self.ttl
        return None

    @property
    def time_left(self):
        if self.started_at and self.ttl:
            now = self.env.now
            shutdown_at = self.shutdown_at
            return shutdown_at - now if now < shutdown_at else 0
        return None

    def run(self, ttl, callback: Callable):
        self.ttl = ttl
        self.started_at = self.env.now
        try:
            yield self.env.timeout(ttl)
            log.debug("CFS in service, running for %d" % ttl)
            callback(self)
        except simpy.Interrupt:
            pass


class DumbAttacker(object):
    """
    Dumb Attacker is an attacker who attacks with a constant probability
    """

    def __init__(self, env: simpy.Environment, cfs_in_service: simpy.Store) -> None:
        super().__init__()
        self.env = env
        self.cfs_in_service = cfs_in_service
        self.successful_attacks = simpy.Store(env)
        self.failed_attacks = simpy.Store(env)
        self.attack_probability = INITIAL_ATTACK_PROBABILITY
        self.min_usable_ttl = ATTACKER_MIN_USABLE_TTL
        self.successful_attack_times = pd.Series()

    @property
    def num_successful_attacks(self) -> int:
        return len(self.successful_attacks.items)

    @property
    def num_failed_attacks(self) -> int:
        return len(self.failed_attacks.items)

    def was_successfully_attacked(self, cfs) -> bool:
        """
        Was a CFS attacked ?
        :param cfs: 
        :return: bool
        """
        return cfs in self.successful_attacks.items

    def _attack_succeded(self, cfs):
        log.debug("Attack successful")
        self.successful_attacks.put(cfs)
        self.successful_attack_times = self.successful_attack_times.append(pd.Series(self.env.now))

    def _attack_failed(self, cfs):
        log.debug("Attack failed")
        self.failed_attacks.put(cfs)

    def _was_attack_successful(self, cfs):
        is_cfs_in_service = cfs in self.cfs_in_service.items
        usable_time = cfs.time_left
        return is_cfs_in_service and usable_time > self.min_usable_ttl

    def run(self):
        while True:
            yield self.env.timeout(1)
            if _decision(self.attack_probability):
                recon_time = random.randint(2, 12)
                log.debug("Recon time...", recon_time)
                yield self.env.timeout(recon_time)

                attack_time = random.randint(4, 12)
                log.debug("Attacking for ... ", attack_time)
                cfs_of_choice = random.choice(self.cfs_in_service.items)
                yield self.env.timeout(attack_time)
                if self._was_attack_successful(cfs_of_choice):
                    self._attack_succeded(cfs_of_choice)
                else:
                    self._attack_failed(cfs_of_choice)


class CloudSystem(object):
    def __init__(self, env: simpy.Environment):
        super().__init__()
        self.env = env
        self.cfs_pool = simpy.Store(env, MAX_POOL_SIZE)
        self.cfs_in_service = simpy.Store(env, MAX_CFS_IN_SERVICE)
        self.attacker = DumbAttacker(env, self.cfs_in_service)
        self.env.process(self.producer())
        self.env.process(self.consumer())
        self.env.process(self.attacker.run())
        self.successful_attacks = []
        self.failed_attacks = []
        self.current_ttl = INITIAL_TTL
        self.scan_probability = INITIAL_SCAN_PROBABILITY
        self.attack_probability = INITIAL_ATTACK_PROBABILITY
        self.discovered_attack_times = pd.Series()
        self.is_adaptive = IS_ADAPTIVE

    def producer(self):
        """
        Adds to the reserve pool
        :return: 
        """
        while True:
            yield self.env.timeout(1)
            while len(self.cfs_pool.items) < self.cfs_pool.capacity:
                self.cfs_pool.put(CFS(self.env))
                log.debug('Adding a new CFS at ', self.env.now)
                yield self.env.timeout(1)

    def consumer(self):
        """
        Consume from reserve pool and bring into service
        :return: 
        """
        while True:
            log.debug("# of CFS in Service", len(self.cfs_in_service.items))
            if len(self.cfs_in_service.items) < self.cfs_in_service.capacity:
                while len(self.cfs_in_service.items) < self.cfs_in_service.capacity:
                    log.debug("Requesting CFS at ", self.env.now)
                    cfs = yield self.cfs_pool.get()
                    self.cfs_in_service.put(cfs)
                    self.env.process(cfs.run(self.current_ttl, lambda c: self.remove_cfs(cfs=c)))
                    yield self.env.timeout(1)
            yield self.env.timeout(1)

    def remove_cfs(self, cfs):
        self.cfs_in_service.items.remove(cfs)
        self.env.process(self.scan(cfs))

    @property
    def attack_rate_based_on_scans(self):
        return self.discovered_attack_times.diff(1).mean()

    def adapt(self):
        """
        Adapt defender by increasing or decreasing ttl
        :return: 
        """
        if self.is_adaptive:
            if self.attack_rate_based_on_scans > ATTACK_THRESHOLD and self.current_ttl > MIN_TTL:
                self.current_ttl -= 1
                self.scan_probability += 0.5
            else:
                self.current_ttl += 1

    def _attack_discovered(self, cfs):
        self.discovered_attack_times = self.discovered_attack_times.append(pd.Series(self.env.now))

    def scan(self, cfs):
        if _decision(self.scan_probability):
            yield self.env.timeout(5)
            if self.attacker.was_successfully_attacked(cfs):
                self._attack_discovered(cfs)
                self.adapt()


def simulate(sim_time: int, ttl: int) -> tuple:
    env = simpy.Environment()
    cloud_system = CloudSystem(env)
    cloud_system.current_ttl = ttl
    env.run(until=sim_time)
    print("Sim ttl ", ttl, " done")
    return (ttl,
            cloud_system.attacker.num_successful_attacks,
            cloud_system.attacker.num_failed_attacks,
            cloud_system.discovered_attack_times.count(),
            sim_time)


def plot(ttl: Iterable, successful_attacks: Iterable, failed_attacks: Iterable, discovered_attacks: Iterable, ):
    plt.plot(ttl, successful_attacks, label='Number of Successful Attacks')
    plt.title('%s adaptive' % ('with' if IS_ADAPTIVE else 'without'))
    plt.plot(ttl, discovered_attacks, label='Number of Discovered Attacks')
    plt.plot(ttl, failed_attacks, label='Number Failed Attacks')
    plt.xlabel('TTL', fontsize='large')
    plt.style.use('presentation')
    plt.ylabel('Number of Attacks')
    plt.legend(prop={'size': 15})

    plt.savefig('%s_adaptive.png' % ('with' if IS_ADAPTIVE else 'without'), bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    simulation_results = []
    raw_results = []
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        simulation_results = [pool.apply_async(simulate, (SIM_TIME, ttl))
                              for ttl in range(1, SIM_TIME, 50)]

        for sim_result in simulation_results:
            initial_ttl, successful_attacks, failed_attacks, discovered_attacks, sim_time = sim_result.get()
            raw_results.append([
                initial_ttl,
                successful_attacks,
                failed_attacks,
                discovered_attacks,
                sim_time
            ])
    transposed = np.array(raw_results).T.tolist()
    plot(transposed[0], transposed[1], transposed[2], transposed[3])

    with open('simulation_results.csv', 'w') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(
                ['Final TTL', '# Successful Attacks', '# Failed Attacks', '# Discovered Attacks', 'Simulation Time'])
        csv_writer.writerows(raw_results)
