import numpy as np
import time


def coulomb_nearest_neighbor_hartree_decomposition(V1, site_densities, nearest_neighbors_hash_table):
    coulomb_terms = np.zeros(len(site_densities), dtype=np.float64)
    for i in range(len(site_densities)):
        coulomb_terms[i] = np.sum(site_densities[nearest_neighbors_hash_table[i]] - 0.5)
    return np.diag(V1 * coulomb_terms)


def hubbard_onsite_hartree_decomposition(U, site_densities):
    if (len(site_densities) % 2) != 0:
        raise ValueError
    total_num_sites = int(len(site_densities) / 2)
    hubbard_terms = np.zeros(len(site_densities), dtype=np.float64)
    hubbard_terms[:total_num_sites] = site_densities[total_num_sites:] - 0.5
    hubbard_terms[total_num_sites:] = site_densities[:total_num_sites] - 0.5
    return np.diag(U * hubbard_terms)


def self_consistent_calculate_coulomb(tbham, nn_hash_table, V1, schf_initial_guess, schf_tolerance, verbose=False, check=False):
    site_occupations = schf_initial_guess
    prev_site_occupations = np.copy(site_occupations)
    check_cond = np.array([])
    convergence_check = True
    while convergence_check:
        start_time = time.time()
        hamiltonian = tbham + coulomb_nearest_neighbor_hartree_decomposition(V1, site_occupations, nn_hash_table)
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
        fill_mask = (eigenvalues < 0)
        site_occupations = np.real(np.sum(eigenvectors[:, fill_mask] * eigenvectors[:, fill_mask].conj(), axis=1))
        convergence_check = np.any(np.abs(site_occupations - prev_site_occupations) > schf_tolerance)
        if verbose:
            print('Iteration took {} seconds with maximum difference {}'.format(time.time() - start_time,
                                                                                np.max(np.abs(site_occupations - prev_site_occupations))))
        if check:
            check_calc = np.abs(np.sum(site_occupations - 0.5))
            check_cond = np.append(check_cond, check_calc)
        prev_site_occupations = np.copy(site_occupations)
    if check:
        return site_occupations, check_cond
    else:
        return site_occupations


def self_consistent_calculate_hubbard(tbham, U, schf_initial_guess, schf_tolerance, verbose=False, check=False,
                                      stuck_check=False):
    site_occupations = schf_initial_guess
    prev_site_occupations = np.copy(site_occupations)
    check_cond = np.array([])
    convergence_check = True
    iteration_counter = 0
    while convergence_check:
        start_time = time.time()
        hamiltonian = tbham + hubbard_onsite_hartree_decomposition(U, site_occupations)
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
        fill_mask = (eigenvalues < 0)
        site_occupations = np.real(np.sum(eigenvectors[:, fill_mask] * eigenvectors[:, fill_mask].conj(), axis=1))
        convergence_check = np.any(np.abs(site_occupations - prev_site_occupations) > schf_tolerance)
        if verbose:
            print('Iteration took {} seconds with maximum difference {}'.format(time.time() - start_time,
                                                                                np.max(np.abs(site_occupations - prev_site_occupations))))
        if check:
            check_calc1 = np.abs(np.sum(site_occupations[:int(len(site_occupations) / 2)] - 0.5))
            check_calc2 = np.abs(np.sum(site_occupations[int(len(site_occupations) / 2):] - 0.5))
            check_cond = np.append(check_cond, np.max(np.array([check_calc1, check_calc2])))
        prev_site_occupations = np.copy(site_occupations)
        iteration_counter += 1
        if (iteration_counter > 150) and stuck_check and convergence_check:
            return None
    if check:
        return site_occupations, check_cond
    else:
        return site_occupations


def delta_cdw_from_occupations(site_occupations):
    delta_a = site_occupations[:int(len(site_occupations) / 2)] - 0.5
    delta_b = site_occupations[int(len(site_occupations) / 2):] - 0.5
    return (np.average(delta_a) - np.average(delta_b)) / 2


def delta_afm_from_occupations(site_occupations):
    delta_a_up = site_occupations[:int(len(site_occupations) / 4)] - 0.5
    delta_b_up = site_occupations[int(len(site_occupations) / 4):int(len(site_occupations) / 2)] - 0.5
    delta_a_down = site_occupations[int(len(site_occupations) / 2):int(3 * len(site_occupations) / 4)] - 0.5
    delta_b_down = site_occupations[int(3 * len(site_occupations) / 4):] - 0.5
    return (np.average(delta_a_up) - np.average(delta_b_up) - np.average(delta_a_down) + np.average(delta_b_down)) / 2


def delta_cdw_from_occupations_center_plaq(site_occupations, pval):
    delta_a = site_occupations[:int(pval / 2)] - 0.5
    delta_b = site_occupations[int(len(site_occupations) / 2):int(len(site_occupations) / 2 + pval / 2)] - 0.5
    return (np.average(delta_a) - np.average(delta_b)) / 2


def delta_afm_from_occupations_center_plaq(site_occupations, pval):
    delta_a_up = site_occupations[:int(pval / 2)] - 0.5
    delta_b_up = site_occupations[int(len(site_occupations) / 4):int(len(site_occupations) / 4 + pval / 2)] - 0.5
    delta_a_down = site_occupations[int(len(site_occupations) / 2):int(len(site_occupations) / 2 + pval / 2)] - 0.5
    delta_b_down = site_occupations[int(3 * len(site_occupations) / 4):int(3 * len(site_occupations) / 4 + pval / 2)] - 0.5
    return (np.average(delta_a_up) - np.average(delta_b_up) - np.average(delta_a_down) + np.average(delta_b_down)) / 2


def delta_cdw_from_occupations_bernalbilayer(site_occupations):
    delta_a_l1 = site_occupations[:int(len(site_occupations) / 4)] - 0.5
    delta_b_l1 = site_occupations[int(len(site_occupations) / 4):int(len(site_occupations) / 2)] - 0.5
    delta_a_l2 = site_occupations[int(len(site_occupations) / 2):int(3 * len(site_occupations) / 4)] - 0.5
    delta_b_l2 = site_occupations[int(3 * len(site_occupations) / 4):] - 0.5
    return (np.average(delta_a_l1) - np.average(delta_b_l1) + np.average(delta_a_l2) - np.average(delta_b_l2)) / 4


def delta_afm_from_occupations_bernalbilayer(site_occupations):
    delta_a_up_l1 = site_occupations[:int(len(site_occupations) / 8)] - 0.5
    delta_b_up_l1 = site_occupations[int(len(site_occupations) / 8):int(len(site_occupations) / 4)] - 0.5
    delta_a_up_l2 = site_occupations[int(len(site_occupations) / 4):int(3 * len(site_occupations) / 8)] - 0.5
    delta_b_up_l2 = site_occupations[int(3 * len(site_occupations) / 8):int(len(site_occupations) / 2)] - 0.5

    delta_a_down_l1 = site_occupations[int(len(site_occupations) / 2):int(5 * len(site_occupations) / 8)] - 0.5
    delta_b_down_l1 = site_occupations[int(5 * len(site_occupations) / 8):int(3 * len(site_occupations) / 4)] - 0.5
    delta_a_down_l2 = site_occupations[int(3 * len(site_occupations) / 4):int(7 * len(site_occupations) / 8)] - 0.5
    delta_b_down_l2 = site_occupations[int(7 * len(site_occupations) / 8):] - 0.5

    return (np.average(delta_a_up_l1) - np.average(delta_b_up_l1) + np.average(delta_a_up_l2) - np.average(delta_b_up_l2)
            - np.average(delta_a_down_l1) + np.average(delta_b_down_l1) - np.average(delta_a_down_l2) + np.average(delta_b_down_l2)) / 4
