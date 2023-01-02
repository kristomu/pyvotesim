import generators
import random
import ballots
import methods
import copy
import math

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

# Return a copy of the ballot with the
# candidate in question having been moved to the top,
# to model compromising strategy.

def rank_top(ballot, candidate):
	without_cddt = [x for x in ballot if x != candidate]
	return [candidate] + without_cddt

# Ditto, but rank the candidate in question at the
# bottom to model burial.
# TODO: Handle truncated ballots where the candidate is
# unranked.
def rank_bottom(ballot, candidate):
	without_cddt = [x for x in ballot if x != candidate]
	return without_cddt + [candidate]

# Generic transformation functions/interfaces. Prior_winner
# is the candidate who used to win, who we're trying to make
# lose, and candidate_to_win is the candidate we're trying to
# get to win.
def compromise_on_ballot(ballot, prior_winner, candidate_to_win):
	return rank_top(ballot, candidate_to_win)

def bury_on_ballot(ballot, prior_winner, candidate_to_win):
	return rank_bottom(ballot, prior_winner)

# Put the prior_winner last and the compromise candidate first
def two_sided_strategy_ballot(ballot, prior_winner,
	candidate_to_win):

	return rank_top(rank_bottom(ballot, prior_winner),
		candidate_to_win)

# Attempt a strategy given by the modify_ballot method.
# It returns None if either the candidate to make win is
# already a winner or the strategy doesn't work, otherwise
# it returns the modified election where strategy worked.

# TODO? Use logging instead of verbose?

def attempt_particular_strategy(election, numcands,
		prior_winner, candidate_to_win, method,
		modify_ballot, verbose=False):
	# First get the initial winners.
	social_order = method(election, numcands)
	winners = set(social_order[0])

	if verbose:
		print("The winner set is", winners)
		print("The social order is", social_order)

	if candidate_to_win in winners or \
		prior_winner not in winners:
		return None

	modified_ballots = copy.copy(election)

	# This can be cached if we're doing multiple
	# strategy types...
	pairwise_support = ballots.get_pairwise_support(
		election, numcands)

	for ballot_idx in pairwise_support[(candidate_to_win,
		prior_winner)]:
		modified_ballots[ballot_idx] = modify_ballot(
			modified_ballots[ballot_idx],
			prior_winner, candidate_to_win)

	social_order_after = method(modified_ballots,
		numcands)
	winners_after = set(social_order_after[0])

	if candidate_to_win in winners_after:
		if verbose:
			print("Now the winner is", candidate_to_win)
			print("The modified election:")
			print(ballots.str_election(modified_ballots))
		return modified_ballots

	return None

def attempt_strategy(election, numcands, method,
	modify_ballot=compromise_on_ballot):
	# Get the initial winners.
	social_order = method(election, numcands)
	winners = set(social_order[0])
	losers = set(range(numcands)) - winners

	# For each loser, against each winner, see if it's
	# possible to do the strategy to get that loser elected.

	for prior_winner in winners:
		for candidate_to_win in losers:
			attempt = attempt_particular_strategy(election,
				numcands, prior_winner, candidate_to_win,
				method, modify_ballot)
			if attempt is not None:
				return attempt

	return None

# MONTE CARLO


# Implement the Agresti-Coull confidence interval for binomial prop.
# Wilson is better but also more of a pain to code.
def ag_confidence(successes, total, pvalue=0.05):
	z = scipy.stats.norm.ppf(1 - (pvalue / 2))
	n_tilde = total + z**2
	p_tilde = 1/n_tilde * (successes + (z**2)/2)

	correction_factor = z * math.sqrt(p_tilde/n_tilde * (1-p_tilde))

	return (max(0, p_tilde - z * correction_factor),
		min(p_tilde + z * correction_factor, 1))

# Check if a particular election admits a particular strategy.
# The outcomes parameter is a dictionary where the keys are the
# strategy types and the values are lists where 0 indicates that
# the strategy didn't work and 1 indicates that it did.
def update_strat_results(election, method, outcomes, numcands):
	strategies = (("Compromising", compromise_on_ballot), (
		"Burial", bury_on_ballot), ("Two-sided", two_sided_strategy_ballot))

	for strategy_name, strat in strategies:
		if not strategy_name in outcomes:
			outcomes[strategy_name] = []

		if attempt_strategy(election, numcands,
			method, strat) is not None:
			outcomes[strategy_name].append(1)
		else:
			outcomes[strategy_name].append(0)

# This returns an outcome dictionary as defined above.
def do_monte_carlo(election_space, method, iterations):
	strategies = (("Compromising", compromise_on_ballot), (
		"Burial", bury_on_ballot), ("Two-sided", two_sided_strategy_ballot))

	outcomes = {name: [] for name, strat in strategies}

	for i in range(iterations):
		election = election_space.random()

		for strategy_name, strat in strategies:
			if attempt_strategy(election, election_space.numcands,
				method, strat) is not None:
				outcomes[strategy_name].append(1)
			else:
				outcomes[strategy_name].append(0)

	return outcomes

# Used to compare results with quadelect - that's why election_str
# is taken from the console. It returns an outcomes dictionary.
def get_manipulability(method, numcands):

	outcomes = {}
	election = ballots.input_election() # Get an election from the console
	update_strat_results(election, method, outcomes, numcands)
	return outcomes

def show_simple_monte_carlo(election_space=
	generators.strict_ordinal_elections(29, 3),
	method=methods.plurality, iterations=20000):

	outcomes = do_monte_carlo(election_space, method, iterations)

	for strategy_name, outcome_per_election in outcomes.items():
		successes = sum(outcome_per_election)
		total = len(outcome_per_election)

		ci_low, ci_high = ag_confidence(successes, total)

		print ("%s: %.4f (c.i: %.4f - %.4f)" % (strategy_name,
			successes/total, ci_low, ci_high))

	return strategy_successes, total

# Add Monte Carlo results for burial, comrpomising, and two-sided strategy.
# TODO: If I add more strategies above, do_monte_carlo will presumably
# check all of them. But if I only need compromising, burial, and two-sided
# strats data, that would be a waste. So somehow convey to do_monte_carlo what
# kind of strategy we're interested in -- or sum all the other strategies up as
# "other".
def get_MC_proportions(monte_carlo_results):

	total = len(monte_carlo_results["Compromising"])

	# We're interested in four proportions: how often burial
	# succeeds, compromising succeds, both succeed, and
	# neither of the two does but two-sided does.

	burial_only = 0
	compromise_only = 0
	burial_and_compromise = 0
	two_sided_only = 0

	for i in range(total):
		burial = monte_carlo_results["Burial"][i] == 1
		compromise = monte_carlo_results["Compromising"][i] == 1
		two_sided = monte_carlo_results["Two-sided"][i] == 1

		if burial and not compromise:
			burial_only += 1
		if burial and compromise:
			burial_and_compromise += 1
		if compromise and not burial:
			compromise_only += 1
		if two_sided and not burial and not compromise:
			two_sided_only += 1

	return {"Burial only": burial_only/total,
		"Compromise only": compromise_only/total,
		"Burial and compromise": burial_and_compromise/total,
		"Two-sided only": two_sided_only/total}

def plot_MC_proportions(
	election_space=generators.strict_ordinal_elections(29, 3),
	iterations=40000):

	method_pairs = [("Copeland", methods.copeland),
		("Minmax", methods.minmax), ("fpA-\nsum fpC", methods.fpA_sum_fpC),
		("fpA-\nmax fpC", methods.fpA_max_fpC), ("Contingent", methods.contingent_vote),
		("Smith-\nContingent", methods.smith_contingent)]

	colors = {"Burial only": "#FF6666", "Compromise only": "#6681FF",
		"Burial and compromise": "#FF66FF", "Two-sided only": "#4FC54F"}

	bar_order = ["Compromise only", "Burial and compromise", "Burial only",
		"Two-sided only"]

	method_names = [name for name, method in method_pairs]

	results_by_method = {}

	for name, method in method_pairs:
		monte_carlo_results = do_monte_carlo(election_space,
			method, iterations)
		results_by_method[name] = get_MC_proportions(monte_carlo_results)
		print(name, results_by_method[name])

	r = range(len(method_names))

	bar_start_points = np.array([0.0] * len(method_names))

	fig, ax = plt.subplots()

	for strategy_name in bar_order:
		strat_success_per_method = [results_by_method[method_name][strategy_name]
			for method_name in method_names]
		ax.bar(r, strat_success_per_method, color=colors[strategy_name],
			edgecolor='white', width=1/np.sqrt(2), label=strategy_name,
			bottom=bar_start_points)
		bar_start_points += strat_success_per_method

	ax.set_ylim([0, 1])

	plt.legend()
	plt.xticks(r, method_names, fontweight='bold')
	plt.ylabel("Strategy susceptibility")
	plt.show()

# Maybe TODO?

# Then later:
# 	exhaustive enumeration, with probabilities according to the model
#		in question,
#   of every election, like in manipulability/fpa_recurs.
#   VSE?