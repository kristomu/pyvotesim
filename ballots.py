# The election format is a list of lists, e.g.
# [[0, 1, 2],
#  [1, 2],
#  [0, 2]]
# is the same as
# A>B>C
# B>C
# A>C.

import numpy as np
import functools
from scipy.sparse.csgraph import connected_components

# This only works for candidate numbers < 26. I'll add more if
# required later.
def str_ballot(ballot):
	return ">".join(
		[chr(ord('A')+ cand_number) for cand_number in ballot])

def str_election(election):
	# First get the counts per ballot type.
	counts_per_type = {}

	for ballot in election:
		if not tuple(ballot) in counts_per_type:
			counts_per_type[tuple(ballot)] = 0
		counts_per_type[tuple(ballot)] += 1

	out = "\n".join([
		"%d: %s" % (count, str_ballot(ballot)) \
			for ballot, count in counts_per_type.items()])

	return out


# Transforming ballots to auxiliary formats that may be needed
# for some methods.
def condorcet_matrix(election, numcands):
	condmat = np.zeros(shape=(numcands, numcands))

	for ballot in election:
		# Candidates may be truncated, so we need to keep track
		# of which we've seen and then rank them all above everybody
		# who has not been seen.
		seen_candidates = set()

		for i in range(len(ballot)):
			seen_candidates.add(ballot[i])
			for j in range(i+1, len(ballot)):
				condmat[ballot[i]][ballot[j]] += 1

		not_seen_candidates = set(range(numcands)) - seen_candidates
		for candidate in seen_candidates:
			for beaten_candidate in not_seen_candidates:
				condmat[candidate][beaten_candidate] += 1

	return condmat

def pairwise_beats(condmat, a, b):
	return condmat[a][b] > condmat[b][a]

def pairwise_ties(condmat, a, b):
	return condmat[a][b] == condmat[b][a]

def pairwise_beats_or_ties(condmat, a, b):
	return condmat[a][b] >= condmat[b][a]

# Given an election, output a dictionary that maps pairs of
# candidates to a list of all the indices of voters who prefer
# the former candidate to the latter. This is used for strategy
# purposes.

def get_pairwise_support(election, numcands):
	pairwise_support = {}

	for i in range(numcands):
		for j in range(numcands):
			if i != j:
				pairwise_support[(i, j)] = []

	# The code is very similar to the Condorcet matrix one.
	for ballot_idx in range(len(election)):
		seen_candidates = set()

		ballot = election[ballot_idx]
		for i in range(len(ballot)):
			seen_candidates.add(ballot[i])
			for j in range(i+1, len(ballot)):
				pairwise_support[(ballot[i], ballot[j])].append(
					ballot_idx)

		not_seen_candidates = set(range(numcands)) - seen_candidates
		for candidate in seen_candidates:
			for beaten_candidate in not_seen_candidates:
				pairwise_support[(candidate,
					beaten_candidate)].append(ballot_idx)

	return pairwise_support

# Helper functions for compositing rankings, e.g. Smith,X or Smith//X

def get_outcome_from_comparison(numcands, relation):
	outcome_without_ties = sorted(range(numcands),
		key=functools.cmp_to_key(relation))

	outcome = []
	current_preference_rank = [outcome_without_ties[0]]
	for cand in outcome_without_ties[1:]:
		if relation(cand, current_preference_rank[0]) == 0:
			current_preference_rank.append(cand)
		else:
			outcome.append(tuple(current_preference_rank))
			current_preference_rank = [cand]
	outcome.append(tuple(current_preference_rank))

	return tuple(outcome)

def break_ties(order, tiebreak, numcands):
	tie_location = [-1 for _ in range(numcands)]
	order_location = [-1 for _ in range(numcands)]

	for preference_idx in range(len(order)):
		for cand in order[preference_idx]:
			order_location[cand] = preference_idx

	for preference_idx in range(len(tiebreak)):
		for cand in tiebreak[preference_idx]:
			tie_location[cand] = preference_idx

	def relation(x, y):
		order_and_tie_x = (order_location[x], tie_location[x])
		order_and_tie_y = (order_location[y], tie_location[y])

		if order_and_tie_x < order_and_tie_y:
			return -1
		if order_and_tie_x > order_and_tie_y:
			return 1
		return 0

	return get_outcome_from_comparison(numcands, relation)

### More Condorcet stuff

def maximal_component(relation_matrix, n):
	num_components, connected = connected_components(relation_matrix,
		connection='strong')

	# One of the components is maximal: find out which
	candidate_in_component = [-1] * num_components

	for i in range(n):
		candidate_in_component[connected[i]] = i

	maximal_set_component = 0
	for i in range(len(candidate_in_component)):
		incumbent = candidate_in_component[maximal_set_component]
		challenger = candidate_in_component[i]
		if relation_matrix[challenger][incumbent] and not \
			relation_matrix[incumbent][challenger]:
			maximal_set_component = i

	maximal_set = [cand for cand in range(n) if \
		connected[cand] == maximal_set_component]

	return maximal_set

def smith_from_cm(condorcet_matrix):
	n = len(condorcet_matrix)
	beats_or_ties = np.zeros(shape=(n, n), dtype=bool)

	for i in range(n):
		for j in range(n):
			beats_or_ties[i][j] = condorcet_matrix[i][j] >= condorcet_matrix[j][i]

	return maximal_component(beats_or_ties, n)

def schwartz_from_cm(condorcet_matrix):
	n = len(condorcet_matrix)
	beats = np.zeros(shape=(n, n), dtype=bool)

	for i in range(n):
		for j in range(n):
			beats[i][j] = condorcet_matrix[i][j] > condorcet_matrix[j][i]

	rating = [True for i in range(n)]
	for i in range(n):
		for j in range(n):
			if i != j and not beats[i][j] and beats[j][i]:
				rating[i] = False

	schwartz_set = [i for i in range(n) if rating[i]]

	# n-way perfect tie, return Smith set.
	if len(schwartz_set) == 0:
		return smith_from_cm(condorcet_matrix)

	return [i for i in range(n) if rating[i]]

def landau_from_cm(condorcet_matrix):
	n = len(condorcet_matrix)
	uncovered = []
	# https://en.wikipedia.org/wiki/Landau_set
	# Lazy O(n^3) approach.
	for x in range(n):
		x_uncovered = True

		for z in range(n):
			if not x_uncovered: continue
			if x == z: continue

			x_covers_z = False

			for y in range(n):
				if x_covers_z: continue
				if not condorcet_matrix[x][y] >= condorcet_matrix[y][x]: continue
				if not condorcet_matrix[y][z] >= condorcet_matrix[z][y]: continue
				x_covers_z = True

			x_uncovered &= x_covers_z

		if x_uncovered:
			uncovered.append(x)

	return uncovered

def smith(election, numcands):
	return smith_from_cm(condorcet_matrix(election, numcands))