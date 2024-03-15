import random

import numpy as np
import pandas as pd
import simpy

from sims.base import _decision

SIM_TIME = 434 * 24

# TTL is time to live
INITIAL_TTL = SIM_TIME
INITIAL_SCAN_PROBABILITY = 0.5
INITIAL_SCAN_TIME = 20
INITIAL_ATTACK_PROBABILITY = 0.10
SERVER_POOL_SIZE = 100  # Number of servers in service
LAMBDA_ATTACKER_RECON_TIME = lambda: np.random.randint(24, 10 * 24)
LAMBDA_ATTACKER_ATTACK_TIME = lambda: np.random.randint(48, 20 * 24)
ATTACKER_MIN_USABLE_TTL = 0.05 * SIM_TIME
ADAPTIVE = False


class CFS(object):
    def __init__(self, env, ttl) -> None:
        super().__init__()
        self.env = env
        self.ttl = ttl
        self.is_compromised = False
        self.started_service_at = env.now

    @property
    def ttl_expired(self):
        return self.env.now - self.started_service_at >= self.ttl

    @property
    def shutdown_at(self):
        if self.started_service_at and self.ttl:
            return self.started_service_at + self.ttl
        return None

    @property
    def time_left(self):
        if self.started_service_at and self.ttl:
            now = self.env.now
            shutdown_at = self.shutdown_at
            return shutdown_at - now if now < shutdown_at else 0
        return None


class DumbAttacker(object):
    def __init__(self, env: simpy.Environment, cfs_in_service: simpy.Store) -> None:
        super().__init__()
        self.env = env
        self.attack_probability = INITIAL_ATTACK_PROBABILITY
        self.cfs_in_service = cfs_in_service
        self.lambda_attacker_recon_time = LAMBDA_ATTACKER_RECON_TIME
        self.lambda_attacker_attack_time = LAMBDA_ATTACKER_ATTACK_TIME
        self.successful_attack_count = 0
        self.attacker_min_usable_time = ATTACKER_MIN_USABLE_TTL
        self.cumulative_usable_time = pd.Series()

    def _was_attack_successful(self, cfs: CFS):
        return ((not (cfs.ttl_expired or cfs.is_compromised))
                and cfs.time_left >= self.attacker_min_usable_time)

    def _attack_succeded(self, cfs: CFS):
        cfs.is_compromised = True
        self.successful_attack_count += 1
        self.cumulative_usable_time = self.cumulative_usable_time.append(
                pd.Series(data=[cfs.time_left], index=[self.env.now]))

    def _attack_failed(self, cfs):
        pass

    def run(self):
        while True:
            yield self.env.timeout(1)
            if _decision(self.attack_probability) and len(self.cfs_in_service.items) > 0:
                yield self.env.timeout(self.lambda_attacker_recon_time())
                yield self.env.timeout(self.lambda_attacker_attack_time())
                cfs_of_choice = random.choice(self.cfs_in_service.items)
                if self._was_attack_successful(cfs_of_choice):
                    self._attack_succeded(cfs_of_choice)
                else:
                    self._attack_failed(cfs_of_choice)


class CloudSystem(object):
    def __init__(self, env: simpy.Environment) -> None:
        super().__init__()
        self.env = env
        self.current_ttl = INITIAL_TTL
        self.scan_probability = INITIAL_SCAN_PROBABILITY
        self.scan_time = INITIAL_SCAN_TIME
        self.server_pool_size = SERVER_POOL_SIZE
        self.in_service_cfs = simpy.Store(env, capacity=self.server_pool_size)
        self.compromise_count = 0
        self.compromise_count_over_time = pd.Series([0], [0], name="Detected compromises over time")
        self.adaptive = ADAPTIVE

    def load_balancer(self):
        """
        Maintains self.server_pool_size number of servers in service
        :return: 
        """

        for cfs in self.in_service_cfs.items:
            if cfs.ttl_expired:
                self.in_service_cfs.items.remove(cfs)
                self.env.process(self.scan(cfs))

        if len(self.in_service_cfs.items) < self.server_pool_size:
            cfs_needed = self.server_pool_size - len(self.in_service_cfs.items)
            for i in range(cfs_needed):
                cfs = CFS(self.env, self.current_ttl)
                self.in_service_cfs.put(cfs)

    def mark_compromised(self, cfs):
        self.compromise_count += 1
        self.compromise_count_over_time = self.compromise_count_over_time.append(
                pd.DataFrame(data=[self.compromise_count], index=[self.env.now]))

    def adapt(self):
        if self.adaptive:
            self.current_ttl *= 0.1

    def scan(self, cfs: CFS):
        if _decision(self.scan_probability):
            yield self.env.timeout(self.scan_time)
            if cfs.is_compromised:
                self.mark_compromised(cfs)
                self.adapt()

    def run(self):
        while True:
            yield self.env.timeout(1)
            self.load_balancer()
