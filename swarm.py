import numpy as np
import numpy.random as random
import math
import scipy.optimize
from time import time
from matplotlib import pyplot as plt
from operator import add

def average_loudness(bats, pop_size):
	loudness = [b["loudness"] for b in bats]
	return sum(loudness)/len(loudness)

def calc_pulse_rate(initial_rate, gamma, iteration):
	return initial_rate * (1 - math.exp(-1 * gamma * iteration))

def position_within_bounds(pos, search_space):
	new = np.copy(pos)
	for dimension, value in enumerate(new):
		if value < search_space[dimension][0]:
			new[dimension] = search_space[dimension][0]
		elif value > search_space[dimension][1]:
			new[dimension] = search_space[dimension][1]
	return new

def local_search(bats, population_count, problem_size, new_position, best, search_space):
	avg_loudness = average_loudness(bats, population_count)     # used to calculate new position
	npos = best["position"] + np.array([random.uniform(-1, 1)*avg_loudness for i in range(problem_size)])
	return position_within_bounds(npos, search_space)

def global_search(bats, problem_size, new_position, best, search_space, i):
	bats[i]["velocity"] =   bats[i]["velocity"] + \
							(bats[i]["position"] - best["position"]) * bats[i]["frequency"]
	pos = bats[i]["position"] + bats[i]["velocity"]
	return position_within_bounds(pos, search_space)

def init_passed_population(population, pop_size, problem_size, freq_min, freq_max, objective):
	pop = [{"position": np.array(population[i]),
			 "velocity": np.array([0]*problem_size),
			 "frequency": random.uniform(freq_min, freq_max),
			 "init_pulse_rate": random.random(),
			 "pulse_rate": 0.0,
			 "loudness": random.uniform(1.0, 2.0)   # "can typically be [1, 2]"
			} for i in range(pop_size)]
	for i in range(pop_size):
		pop[i]["fitness"] = objective(pop[i]["position"])
	return pop

def search(objective, search_space, max_generations, population, 
		   freq_min=0.0, freq_max=1.0, alpha=0.9, gamma=0.9):
	problem_size = len(search_space)    # search space provides bounds for each dimension of problem,
										# length of this list provides number of dimensions
	# initialize bat population using passed population
	population_count = len(population)
	bats = init_passed_population(population, population_count, problem_size, freq_min, freq_max, objective)
	new_position = np.array([0]*problem_size)     # list to hold created candidate solutions
	best = min(bats, key=lambda x:x["fitness"]).copy()         # store intial best bat, based on lowest fitness
	# main loop runs for specified number of iterations
	for t in range(max_generations):
		# loop over all bats in population
		for i in range(population_count):
			# generate new solutions by adjusting frequency, updating velocities and positions
			# calculate new frequency for bat, uniform random between min and max
			bats[i]["frequency"] = random.uniform(freq_min, freq_max)
			new_position = global_search(bats, problem_size, new_position, best, search_space, i)
			if (random.random() > bats[i]["pulse_rate"]):
				# generate local solution around selected best solution
				new_position = local_search(bats, population_count, problem_size, new_position, best, search_space)
			new_fitness = objective(new_position)   # evaluate fitness of new solution
			# new solution position replaces current bat if it has lower fitness
			# AND a random value [0, 1) is less than current loudness
			if (random.random() < bats[i]["loudness"] and new_fitness < bats[i]["fitness"]):
				bats[i]["position"] = new_position.copy()           # accept new solution
				bats[i]["fitness"] = new_fitness
				bats[i]["loudness"] = alpha * bats[i]["loudness"]   # update bat loudness
				bats[i]["pulse_rate"] = calc_pulse_rate(bats[i]["init_pulse_rate"], gamma, t)   # calculate pulse rate to be used in conditional
			# if new generated solution has better fitness than previous best, it becomes new best
			if (new_fitness < best["fitness"]):
				best["position"] = new_position.copy()
				best["fitness"] = new_fitness
	# return list of final position vectors
	final_positions = [bats[i]["position"] for i in range(population_count)]
	#if __name__== "__main__":
		#print("best =", best["position"], "fitness =", best["fitness"])    # un-comment to print out results
	return best
	#return final_positions


def Rastrigin(vec):
	A = 10
	n = len(vec)
	return A*n + np.sum([x**2 - A*np.cos(2*np.pi*x) for x in vec])

def RastriginRand(pop_size, dimensions):
	out_pop = []
	for i in range(pop_size):
		out_pop.append([np.random.uniform(-5.12, 5.12) for i in range(dimensions)])
		
	return out_pop


def sphere(vec):
	return np.sum([x**2 for x in vec]) + 1

def main():

	gens = 6
	pop = 6
	freq_min = 0.046744
	freq_max = 0.922789
	alpha = 0.825152
	gamma = 0.755835
	objective_function = Rastrigin
	estimate_function = RastriginRand
	useswarm = True

	runs = 100
	dimmin = 2
	dimmax = 20

	all_candidates = []
	all_swtimes = []
	all_optimes = []
	all_fracs = []

	if True:	# pre-recorded (another 2 runs) of swarm / no swarm just optimization time for comparison
		wswarm = [0.0012740349769592284, 0.0018128705024719237, 0.002379629611968994, 0.003147881031036377, 0.004097788333892822, 0.005007147789001465, 0.0058867716789245605, 0.00691429853439331, 0.007924890518188477, 0.009407808780670166, 0.010418932437896728, 0.012112195491790772, 0.01305190086364746, 0.014059579372406006, 0.016568405628204344, 0.018151302337646485, 0.019293146133422853, 0.02224048614501953, 0.02256796360015869]
		noswarm = [0.0014391136169433594, 0.0020290064811706544, 0.002706201076507568, 0.003460264205932617, 0.004162876605987549, 0.0052107834815979005, 0.006288037300109863, 0.007382235527038574, 0.008549962043762207, 0.010081918239593505, 0.010951273441314698, 0.012684488296508789, 0.014555959701538087, 0.01631117343902588, 0.01852898597717285, 0.019497482776641845, 0.021971554756164552, 0.02350996494293213, 0.025964550971984864]plt.plot(list(range(dimmin,dimmax+1)), noswarm, label='SLSQP Time, no swarm')
		plt.plot(list(range(dimmin,dimmax+1)), wswarm, label='SLSQP Time, w/ swarm')
		plt.title("Numerical optimization time comparison")
		plt.xlabel("Dimensions"); plt.ylabel("Time(s)")
		plt.grid(alpha=0.25)
		plt.legend()
		plt.xticks(np.arange(dimmin, dimmax+1, 3))
		plt.show()

	tstart = time()
	for dim in range(dimmin, dimmax+1):
		print(dim)

		successes = 0
		candidates = []

		swtimes = []
		optimes = []
		fracs = []

		for run in range(runs):

			search_space = [[-5.12, 5.12] for x in range(dim)]
			init_pop = estimate_function(pop, dim)
			dstart = time()

			if useswarm:
				best = search(objective_function, search_space, gens, init_pop,
					freq_min, freq_max, alpha, gamma)
			else:
				#print(init_pop)
				best = {"position":min(init_pop, key = lambda x: objective_function(x))}
				dstart = time()


			swtime = time() - dstart

			r = scipy.optimize.minimize(objective_function, best["position"], method='SLSQP', tol=1e-10, options={'maxiter':100})

			opttime = time() - dstart - swtime

			if r.success:
				successes += 1
			sol = list(r.x)
			fracs.append(swtime/(swtime + opttime))
			candidates.append(sol)
			swtimes.append(swtime)
			optimes.append(opttime)

		print(successes, end = ",")

		all_fracs.append(fracs)
		all_swtimes.append(swtimes)
		all_candidates.append(candidates)
		all_optimes.append(optimes)

	print()



	avg_deviation = [sum([objective_function(x) for x in dim_list])/len(dim_list) for dim_list in all_candidates]
	plt.plot(list(range(dimmin,dimmax+1)), avg_deviation)
	plt.title(f"Score vs Dimensions (avg. {runs} runs, {'w/ swarm' if useswarm else 'no swarm'})"); plt.xlabel("Dimensions"); plt.ylabel("Rastrigin Score")
	plt.xticks(np.arange(dimmin, dimmax+1, 3))
	plt.grid(alpha=0.25)
	plt.show()

	if True:
		if(useswarm):
			avg_swtime = [sum(dim_list)/len(dim_list) for dim_list in all_swtimes]
			avg_optime = [sum(dim_list)/len(dim_list) for dim_list in all_optimes]
			avg_tottime = list( map(add, avg_swtime, avg_optime) )
			print(avg_optime)
			plt.plot(list(range(dimmin,dimmax+1)), avg_tottime, label='Total Time')
			plt.plot(list(range(dimmin,dimmax+1)), avg_swtime, label='Swarm Time')
			plt.plot(list(range(dimmin,dimmax+1)), avg_optime, label='SLSQP Time')
			plt.legend()
			plt.title(f"Time vs Dimensions (avg. {runs} runs, {'w/ swarm' if useswarm else 'no swarm'})"); plt.xlabel("Dimensions"); plt.ylabel("Time(s)")
			plt.xticks(np.arange(dimmin, dimmax+1, 3))
			plt.grid(alpha=0.25)
			plt.show()
		else:
			avg_tottime = [sum(dim_list)/len(dim_list) for dim_list in all_optimes]
			print(avg_tottime)
			plt.plot(list(range(dimmin,dimmax+1)), avg_tottime, label='Total Time')
			plt.legend()
			plt.title(f"Time vs Dimensions (avg. {runs} runs, {'w/ swarm' if useswarm else 'no swarm'})"); plt.xlabel("Dimensions"); plt.ylabel("Time(s)")
			plt.xticks(np.arange(dimmin, dimmax+1, 3))
			plt.grid(alpha=0.25)
			plt.show()

		avg_frac = [sum(dim_list)/len(dim_list) for dim_list in all_fracs]
		plt.plot(list(range(dimmin,dimmax+1)), avg_frac)
		plt.legend()
		plt.title(f"Swarm Time Percentage of Total Time (avg. {runs} runs, {'w/ swarm' if useswarm else 'no swarm'})"); plt.xlabel("Dimensions"); plt.ylabel("Percentage")
		plt.xticks(np.arange(dimmin, dimmax+1, 3))
		plt.grid(alpha=0.25)
		plt.show()


if __name__== "__main__":
	main()
