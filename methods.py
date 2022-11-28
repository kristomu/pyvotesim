import ballots

# Example

# Scores are of the form one list per candidate to support
# leximax type things like Ext-Plurality.
def score_to_ranking(scores, numcands):
	sorted_outcome = [(list(scores[cand]), cand) for cand in \
		range(numcands)]

	sorted_outcome.sort(reverse=True)

	ranking = [[sorted_outcome[0][1]]]

	rank = 0
	for i in range(1, len(sorted_outcome)):
		if sorted_outcome[i][0] != sorted_outcome[i-1][0]:
			ranking.append([])
		ranking[-1].append(sorted_outcome[i][1])

	return ranking

# Should this be somewhere else?
def smith_from_cm(condorcet_matrix):
	n = len(condorcet_matrix)
	beats_or_ties = np.zeros(shape=(n, n), dtype=bool)

	for i in range(n):
		for j in range(n):
			beats_or_ties[i][j] = condorcet_matrix[i][j] >= condorcet_matrix[j][i]

	return maximal_component(beats_or_ties, n)

#---

def plurality_scores(election, numcands):
	first_prefs = [[0] for _ in range(numcands)]

	for ballot in election:
		first_prefs[ballot[0]][0] += 1

	return first_prefs

# This is the version that scores truncated ballots from the bottom.
def borda_scores(election, numcands):
	points = [[0] for _ in range(numcands)]

	for ballot in election:
		for rank in range(len(ballot)):
			# Last rank gives zero points.
			points[ballot[rank]][0] += len(ballot) - rank - 1

	return points

def minmax_scores(election, numcands):
	# Set the default value to the maximum a Condorcet matrix
	# cell may contain so that the count is updated at least once.
	numvoters = len(election)
	slimmest_victory = [[numvoters] for _ in range(numcands)]

	condmat = ballots.condorcet_matrix(election, numcands)

	for incumbent in range(numcands):
		for challenger in range(numcands):
			if challenger != incumbent:
				slimmest_victory[incumbent][0] = min(
					slimmest_victory[incumbent][0],
					condmat[incumbent][challenger])

	return slimmest_victory

def cond_copeland_scores(condorcet_matrix, win, tie):
	numcands = len(condorcet_matrix)
	copeland_score = [[0] for _ in range(numcands)]

	for a in range(numcands):
		for b in range(numcands):
			if a == b: continue
			if ballots.pairwise_beats(condorcet_matrix, a, b):
				copeland_score[a][0] += win
			if ballots.pairwise_ties(condorcet_matrix, a, b):
				copeland_score[a][0] += tie

	return copeland_score

def copeland_scores(election, numcands, win=1, tie=0):
	condmat = ballots.condorcet_matrix(election, numcands)
	return cond_copeland_scores(condmat, win, tie)

def weak_condorcet_winners(condorcet_matrix):
	numcands = len(condorcet_matrix)
	copeland_score = cond_copeland_scores(condorcet_matrix, 1, 1)
	return set([x for x in range(numcands) if copeland_score[x][0] == numcands-1])

def simple_fpA_fpC_scores(election, numcands, aggregator,
	condorcet=False):

	scores = [[0] for _ in range(numcands)]

	condmat = ballots.condorcet_matrix(election, numcands)

	# A's score is A's first preference count minus the sum (or max)
	# of first preferences of candidates beating A pairwise.
	first_prefs = plurality_scores(election, numcands)

	for candidate in range(numcands):
		# The [0] term is required because max() won't accept an empty
		# list.
		scores[candidate][0] = first_prefs[candidate][0] - aggregator(
			[0] + [first_prefs[i][0] for i in range(numcands) if \
			 ballots.pairwise_beats(condmat, i, candidate)])

	# Do Condorcet//fpA-fpC if instructed by penalizing every non-Condorcet
	# candidate by a score worse than the worst possible.
	if condorcet:
		numvoters = len(election)
		best_possible_score = numvoters
		worst_possible_score = -numvoters
		for not_winner in set(range(numcands)) - \
			weak_condorcet_winners(condmat):
			scores[not_winner][0] += worst_possible_score - best_possible_score

	return scores

def plurality(election, numcands):
	return score_to_ranking(plurality_scores(election, numcands), numcands)

def borda(election, numcands):
	return score_to_ranking(borda_scores(election, numcands), numcands)

def minmax(election, numcands):
	return score_to_ranking(minmax_scores(election, numcands), numcands)

def fpA_sum_fpC(election, numcands):
	return score_to_ranking(
		simple_fpA_fpC_scores(election, numcands, sum, True), numcands)

def fpA_max_fpC(election, numcands):
	return score_to_ranking(
		simple_fpA_fpC_scores(election, numcands, max, True), numcands)

def copeland(election, numcands):
	return score_to_ranking(
		copeland_scores(election, numcands), numcands)


# Some variants of the Contingent vote, inspired by JGA's results
# Possibly high-resistance *and* summable (though nonmonotone)

def generic_contingent_vote(election, numcands, base_results):
	condmat = ballots.condorcet_matrix(election, numcands)

	# Then the CV winners are the winners of pairwise comparisons
	# between every candidate ranked first and second by Plurality.
	# (My tiebreak)

	if len(base_results) < 2:
		return base_results # every candidate is tied

	contingent_winners = set()

	for winner in base_results[0]:
		for runner_up in base_results[1]:
			if ballots.pairwise_beats_or_ties(condmat, winner, runner_up):
				contingent_winners.add(winner)
			if ballots.pairwise_beats_or_ties(condmat, runner_up, winner):
				contingent_winners.add(runner_up)

	contingent_losers = set(range(numcands)) - contingent_winners

	return [list(contingent_winners), list(contingent_losers)]

def contingent_vote(election, numcands):
	plur_results = plurality(election, numcands)
	return generic_contingent_vote(election, numcands, plur_results)

def comma(election, set_method, base_method, numcands):
	constraint_set = set(set_method(election, numcands))
	constraint_order = (tuple(constraint_set), tuple(
		set(range(numcands))-constraint_set))
	return ballots.break_ties(constraint_order,
		base_method(election, numcands), numcands)

def slash(election, set_method, base_method, numcands):
	constraint_set = set(set_method(election, numcands))
	everybody_else = set(range(numcands)) - constraint_set
	members = list(constraint_set)

	# Now create a mapping from the candidates in the constraint set
	# to consecutive numbers, and create an eliminated election where
	# every candidate but the ones in the set has been eliminated.
	mapping_to_smaller = {members[i]: i for i in range(len(members))}
	mapping_to_larger = {i: members[i] for i in range(len(members))}

	shrunk_election = []
	for ballot in election:
		shrunk_ballot = [mapping_to_smaller[i] for i in ballot\
			if i in mapping_to_smaller]
		# Don't add exhausted ballots.
		if len(shrunk_ballot) > 0:
			shrunk_election.append(shrunk_ballot)

	smaller_outcome = base_method(shrunk_election, len(members))

	# Translate back
	larger_outcomes = []
	for rank in smaller_outcome:
		larger_outcomes.append([mapping_to_larger[cand] for cand in rank])

	# Everybody not in the set comes last
	larger_outcomes += [list(everybody_else)]

	return larger_outcomes

# Smith-Contingent Vote: Rank candidates by Smith, Plur instead.

def smith_contingent(election, numcands):
	smith_plur_results = comma(election, ballots.smith,
		plurality, numcands)
	return generic_contingent_vote(election, numcands,
		smith_plur_results)

def smslash_contingent(election, numcands):
	return slash(election, ballots.smith, contingent_vote,
		numcands)