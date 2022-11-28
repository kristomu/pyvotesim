import math
import random

# Generators for ballots and elections. These are deliberately
# unoptimized; in particular, iterating over elections will call
# a bunch of from-scratch ballot generation functions. I'll
# optimize later if it's necessary.

# TODO? Weighted models (e.g. IAC) - how? Output probability of picking
# that particular value, like a pdf?
# And perhaps a weighted model like IC but with redundant elections removed
# (e.g. different ballot order, different candidate order)?

# Ballots: n candidates, n! permutations.
def int_to_permutation(idx, num_entries):

	entry_list = list(range(num_entries))
	permutation = []

	# Now extract digits in factorial base.
	for base in range(num_entries, -1, -1):
		unit_value = math.factorial(base)
		element_this_digit = idx // unit_value
		idx %= unit_value

		# The factoradic element is the index into the
		# entry list, after all preceding indices' numbers have
		# been removed, for the next element in the permutation,
		# but the very first element is to absorb the higher
		# "digits" of numbers that are too large.
		if base != num_entries:
			permutation.append(entry_list.pop(element_this_digit))

	return permutation

# Define an iterator that's determined by:
# - having a number of entries function that takes one parameter
# - having a "get nth entry" function.

class iterable_max:
	def __init__(self, numcands):
		self.numcands = numcands
		self.num_entries = self._num_entries(numcands)
		self.cur_entry = 0

	def __len__(self):
		return self.num_entries

	def __iter__(self):
		return self

	def __next__(self):
		if self.cur_entry >= self.num_entries:
			raise StopIteration

		output = self._nth_item(self.cur_entry,
			self.numcands)
		self.cur_entry += 1
		return output

	def __getitem__(self, idx):
		return self._nth_item(idx, self.numcands)

	# For larger election spaces, we can't simply use
	# random.choice since len() will throw an exception
	# if the value is greater than the system maximum.
	# This is a known CPython bug:
	# https://stackoverflow.com/questions/60340710
	def random(self):
		return self[random.randint(0, self.num_entries-1)]

class strict_ordinal_ballots(iterable_max):
	def _num_entries(self, numcands):
		return math.factorial(numcands)

	def _nth_item(self, cur, numcands):
		return int_to_permutation(cur, numcands)

# n!/(n-k)!
def partial_factorial(k, n):
	out = 1
	for i in range(1, n+1):
		if i > n-k:
			out *= i
	return out

#n candidates, k < n ranks, n!/(n-k)! permutations
def int_to_partial_permutation(idx, ranks, num_entries):
	entry_list = list(range(num_entries))
	permutation = []

	unit_value = partial_factorial(ranks, num_entries)

	for base in range(num_entries, num_entries-ranks-1, -1):
		element_this_digit = idx // unit_value
		idx %= unit_value
		if base > 0:
			unit_value //= base

		if base != num_entries:
			permutation.append(entry_list.pop(element_this_digit))

	return permutation

def int_to_truncated_ballot(idx, numcands):
	# Here the idea is that if we know how many candidates the voter
	# ranks before truncating, we could just use
	# int_to_partial_permutation above.

	# Thus to produce a general mapping of integers to truncated
	# ballots, we can concatenate the mappings from integers to
	# ballots that rank one candidate, ballots that rank two, and
	# so on, so that:
	# 0 ...n_1 = some ballot that ranks one candidate
	# n_1 ... n_1 + n_2 = some ballot that ranks two,
	# etc.

	# The ordering is thus something like:
	# 	0 ... n!/(n-1)!		                 one candidate chosen
	#	n!/(n-1)! ... n!/(n-1)! + n!/(n-2)!  two candidates
	# etc.

	# When turning an integer into a ballot, we first have to find
	# out which stack it is in (i.e. how many candidates it ranks),
	# then its position within the stack (which is resolved by
	# int_to_partial_permutation).

	# The closed form expression for \sum i=0...k n!/(n-i)! contains
	# incomplete gamma functions, hence terms of the form floor(e * p!).
	# I don't like mixing ints and floats, so I'm just gonna enumerate
	# the values and make a cumulative list instead.

	no_more_than_k_cddts = [0]

	for k in range(1, numcands+1):
		no_more_than_k_cddts.append(no_more_than_k_cddts[-1] + \
			partial_factorial(k, numcands))

	# Travel up the list to find the greatest value not exceeding
	# the index. If it's the kth value, then the index belongs to
	# the kth stack, which corresponds to ranking k+1 candidates.

	greatest_not_exceeding = 0
	for k in range(1, numcands+1):
		if idx >= no_more_than_k_cddts[k]:
			greatest_not_exceeding = k

	# Subtract the value of every stack below this one so that we
	# get the zero-indiced value. Then pass it through the truncated
	# permutation function.

	return int_to_partial_permutation(idx - \
		no_more_than_k_cddts[greatest_not_exceeding],
		greatest_not_exceeding+1, numcands)

class partial_ordinal_ballots(iterable_max):
	def _num_entries(self, numcands):
		return sum([partial_factorial(k, numcands) \
			for k in range(1, numcands+1)])

	def _nth_item(self, cur, numcands):
		return int_to_truncated_ballot(cur, numcands)


# Now elections, which are really just n ballot generators in
# series. The mapping from integers just uses a positional number
# system where the base is the number of distinct ballots.

# This is the base class.
class iterable_elections:
	def _num_entries(self, numvoters):
		return len(self.base_ballots) ** numvoters

	def __init__(self, numvoters, numcands):
		self.numcands = numcands
		self.numvoters = numvoters
		self.base_ballots = self._set_base_ballot_type(
			numcands)
		self.num_entries = self._num_entries(numvoters)
		self.cur_entry = 0

	def __len__(self):
		return self.num_entries

	def __iter__(self):
		return self

	def __next__(self):
		if self.cur_entry >= self.num_entries:
			raise StopIteration

		output = self[self.cur_entry]
		self.cur_entry += 1
		return output

	def __getitem__(self, idx):
		# Do a bottom-up number conversion because
		# it's less likely to cause overflow problems.
		# This strictly speaking means that the top ballots
		# change first (instead of the bottom as would be
		# expected)

		out = []
		for i in range(self.numvoters):
			this_element = idx % len(self.base_ballots)
			idx //= len(self.base_ballots)
			out.append(self.base_ballots[this_element])

		return out

	def random(self):
		return self[random.randint(0, self.num_entries-1)]

class strict_ordinal_elections(iterable_elections):
	def _set_base_ballot_type(self, numcands):
		return strict_ordinal_ballots(numcands)

class partial_ordinal_elections(iterable_elections):
	def _set_base_ballot_type(self, numcands):
		return partial_ordinal_ballots(numcands)