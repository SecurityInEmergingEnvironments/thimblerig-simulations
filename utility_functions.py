import numpy as np


def monthly_attacker_utility(ttl_hrs=1):
    ec2_cost_per_hr = 0.5
    hrs_in_a_month = 730
    dev_cost_per_hr = 30
    attack_hrs = 0.1

    botnet_size = 100
    value_per_machine = 0.25
    probability_of_attack = 0.99

    botnet_value = botnet_size * value_per_machine

    usable_hrs_per_attack = ttl_hrs - attack_hrs

    if usable_hrs_per_attack > 0:
        attack_count = hrs_in_a_month / usable_hrs_per_attack
    else:
        attack_count = np.inf

    total_dev_cost = (attack_count * attack_hrs) * dev_cost_per_hr

    attack_cost = ec2_cost_per_hr + total_dev_cost
    return botnet_value - (attack_cost * probability_of_attack)


def cost_of_scan(probability_of_scan=0.2):
    automated_scan_hrs = 5  # hrs
    scan_cost_per_hr = 1  # dollars
    automated_scan_total_cost = automated_scan_hrs * scan_cost_per_hr

    manual_scan_hrs = 2
    manual_scan_cost_per_hr = 100
    probability_of_manual_scan = 0.3
    manual_scan_total_cost = manual_scan_hrs * manual_scan_cost_per_hr * probability_of_manual_scan

    return probability_of_scan * (automated_scan_total_cost + manual_scan_total_cost)


def monthly_defender_utility(ttl_hrs=1, probability_of_attack=0.5):
    server_count = 100
    buffer_percent = probability_of_attack / ttl_hrs

    actual_server_count = server_count + (buffer_percent * server_count)

    # TODO tending to infinity when reboot times are too short

    costs_per_server_hr = 0.023
    hrs_in_a_month = 730
    total_server_costs = actual_server_count * costs_per_server_hr * hrs_in_a_month

    probability_of_scan = 0.2
    scan_cost = cost_of_scan(probability_of_scan)

    return scan_cost + total_server_costs
