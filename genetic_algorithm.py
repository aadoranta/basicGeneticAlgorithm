import random
import math
from itertools import permutations
from matplotlib import pyplot as plt


def binary_list(decimal):
    # Function to convert a decimal number (input) into a binary number (output).
    # The binary number takes the form of a list of 1's and 0's of length 12.
    res = [int(i) for i in list('{0:0b}'.format(decimal))]
    zero_length = 12 - len(res)
    zeros = [0] * zero_length
    res = zeros + res
    return res


def weight_value_finder(phenotype):
    # This function takes a phenotype input and using the hard-encoded weight_value list (as per what
    # is given in the instructions), determines the weight and value of the phenotype based on its
    # position of 1's and 0's. As an example, the function will determine the weight of the phenotype
    # by summing all of the weights in the weight_value list (where weights are the 0th entry for every
    # sub-list) for every element corresponding to an element in the phenotype where the phenotype element
    # value is 1. The same is true for value.
    weight = 0
    value = 0
    items = [index for index, element in enumerate(phenotype) if element == 1]
    weight_value = [[20, 6], [30, 5], [60, 8], [90, 7], [50, 6], [70, 9], [30, 4],
                    [30, 5], [70, 4], [20, 9], [20, 2], [60, 1]]
    for item in items:
        weight += weight_value[item][0]
        value += weight_value[item][1]

    return weight, value


def population_generator(size):
    # Functions generates an initial population of size = size where each member is a binary number
    # between 0 and 2^12 represents by a list of 1's and 0's
    population = list()
    values = random.sample(range(2**12 - 1), size)
    for value in values:
        population.append(binary_list(value))
    return population


def fitness_calc(phenotypes):
    # Function takes in phenotypes and determines the fitness of each by finding their respective
    # weight and value. The fitness value for a given phenotype corresponds
    # to its value; however, the fitness becomes 0 if the weight
    # of the phenotype is greater than 250 (the maximum load for the backpack).
    # Fitness function also sorts the list of input phenotypes. The fitness_calc function
    # also determines the average value and best value for an aggregate group of phenotypes in order
    # to compare the best fitness of a population with its average fitness.
    fitness_vales = list()
    for phenotype in phenotypes:
        weight, value = weight_value_finder(phenotype)
        if weight > 250:
            fitness = 0
        else:
            fitness = value
        fitness_vales.append([fitness, phenotype])
    fitness_vales = sorted(fitness_vales, key=lambda x: x[0])
    fitness_vales.reverse()
    values = [item[0] for item in fitness_vales]
    avg_value = sum(values)/len(values)
    best_value = values[0]
    return fitness_vales, avg_value, best_value


def selection(fit_phenotypes, cull):
    # The selection function simply takes the top cull*len(phenotypes) where cull is
    # a number in [0, 1]. It returns the top phenotypes as a list.
    phenotypes = [item[1] for item in fit_phenotypes]
    cull_point = math.floor(len(phenotypes)*cull)
    top_n = phenotypes[:cull_point]
    return top_n


def crossover(top_phenotypes, size):
    # The crossover takes in the top_n phenotypes and mates them. The parents are randomly selected
    # as unique combinations of two numbers within the range of the top_n phenotypes. The new population
    # is returned as the children of the top_n phenotypes.
    values = list(range(len(top_phenotypes)))
    parents = list(permutations(values, 2))
    if size % 2 != 0:
        raise Exception("Population size should be an even number")
    if size < 6:
        raise Exception("Population size must be greater than 6")
    amt_selected = int(size / 2)
    pairs = random.sample(parents, amt_selected)
    children = one_point_crossover(top_phenotypes, pairs)
    new_population = children
    return new_population


def elitism_crossover(phenotypes, size):
    # This function performs exactly the same as the previous crossover function, except that instead of
    # generating a new population that is entirely composed of children of the top_n population, it will
    # retain the top 2 phenotypes from the previous generation.
    values = list(range(len(phenotypes)))
    parents = list(permutations(values, 2))
    if size % 2 != 0:
        raise Exception("Population size should be an even number")
    if size < 6:
        raise Exception("Population size must be greater than 6")
    amt_selected = int(size/2 - 1)
    pairs = random.sample(parents, amt_selected)
    children = one_point_crossover(phenotypes, pairs)
    children.append(phenotypes[0])
    children.append(phenotypes[1])
    new_population = children
    return new_population


def one_point_crossover(phenotypes, pairs):
    # This function takes the top_n phenotypes from the crossover function and finds parents
    # as defined by the pairs list evaluated in the crossover function. It splits two parents at a randomly
    # selected point along their length and recombines the two parents into two children. The two children
    # are therefore a heterogeneous recombination of their parents. Children are then combined into a list
    # and returned.
    children = list()
    for pair in pairs:
        parent1 = phenotypes[pair[0]]
        parent2 = phenotypes[pair[1]]
        cut_point = random.randint(1, len(parent1) - 1)
        cut1_p1 = parent1[:cut_point]
        cut2_p1 = parent1[cut_point:]
        cut1_p2 = parent2[:cut_point]
        cut2_p2 = parent2[cut_point:]
        child1 = cut1_p1 + cut2_p2
        child2 = cut1_p2 + cut2_p1
        children.append(child1)
        children.append(child2)
    return children


def mutation(phenotypes, probability):
    # The mutation function takes in a list of phenotypes and the probability that any given
    # given phenotype will be mutated. An amount (probability*len(phenotypes)) of phenotypes
    # are selected to be mutated. During mutation, a random point on the phenotype is selected
    # and in that position a 0 is changed to a 1 or vice versa. The mutated phenotypes are
    # recombined with the total population and then returned.
    amt_mutated = math.ceil(len(phenotypes) * probability)
    values = list(range(len(phenotypes)))
    selections = random.sample(values, amt_mutated)
    for select in selections:
        mutant = phenotypes[select]
        phenotypes.pop(select)
        mut_point = random.randint(0, len(mutant) - 1)
        if mutant[mut_point] == 1:
            mutant[mut_point] = 0
        elif mutant[mut_point] == 0:
            mutant[mut_point] = 1
        phenotypes.append(mutant)
    return phenotypes


def elitism_mutation(phenotypes, probability):
    # The elitism_mutation performs exactly the same way as the mutation function, except that
    # it preserves the last two phenotypes in the phenotypes list where the last two
    # phenotypes are the phenotypes retained from the previous generation. It similarly returns
    # a list of mutated phenotypes.
    subset_phenotypes = phenotypes[:-2]
    amt_mutated = math.ceil(len(subset_phenotypes) * probability)
    values = list(range(len(subset_phenotypes)))
    selections = random.sample(values, amt_mutated)
    for select in selections:
        mutant = subset_phenotypes[select]
        subset_phenotypes.pop(select)
        mut_point = random.randint(0, len(mutant) - 1)
        if mutant[mut_point] == 1:
            mutant[mut_point] = 0
        elif mutant[mut_point] == 0:
            mutant[mut_point] = 1
        subset_phenotypes.append(mutant)
    subset_phenotypes.append(phenotypes[-2])
    subset_phenotypes.append(phenotypes[-1])
    phenotypes = subset_phenotypes
    return phenotypes


def items_printer(phenotype):
    # Function prints the items (as per the assignment description) that have been selected
    # by the algorithm
    items = [index for index, element in enumerate(phenotype) if element == 1]
    weight_value = [[20, 6], [30, 5], [60, 8], [90, 7], [50, 6], [70, 9], [30, 4],
                    [30, 5], [70, 4], [20, 9], [20, 2], [60, 1]]
    print("--------Items Selected--------")
    for item in items:
        print("Item", item + 1, "-- Weight:", weight_value[item][0], "Value:",
              weight_value[item][1])


def all_possible_pairs(phenotypes):
    # Finds the total euclidean distances between all members of a given generation. Phenotypes
    # represent the complete population for a given generation. Returns the sum of all euclidean distances.
    decimals = list()
    distance_sum = 0
    for phenotype in phenotypes:
        decimals.append(sum(val * (2 ** idx) for idx, val in enumerate(reversed(phenotype))))
    for index in range(len(decimals) - 1):
        for index2 in range(1, (len(decimals))):
            distance_sum += math.sqrt((decimals[index] - decimals[index2]) ** 2)
    return distance_sum


def plt_fitness(fitness, avg_fitness, generation_num, population_size):
    # The plt_fitness function simply tracks the best, and average fitness of a each generation
    # and shows them on a fitness vs. generation number graph.
    plt.plot(fitness, label="Best Fitness Value")
    plt.plot(avg_fitness, label="Average Population Fitness Value")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.suptitle("Fitness Graph")
    plt.title(str("Final Fitness: " + str(fitness[-1]) + ";  Generations: " + str(generation_num)
                  + ";  Pop. Size: " + str(population_size)))
    plt.legend()
    plt.show()


def plt_all_pp_diversity(all_pp_list, fitness):
    # The plt_all_pp_diversity function accepts a list of the sum of all euclidean distances
    # between members over a number of generations and plots these values against the number of
    # generations. It also plots the best fitness value against generations.
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Generations')
    ax1.set_ylabel('Diversity', color=color)
    ax1.plot(all_pp_list, color=color, label="All-Possible-Pairs Diversity")
    ax1.legend(loc='center left', bbox_to_anchor=(1.1, 0.1))

    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.set_ylabel('Fitness', color=color)
    ax2.plot(fitness, color=color, label="Best Fitness Value")
    ax2.legend(loc='center left', bbox_to_anchor=(1.1, 0.2))

    fig.tight_layout()
    plt.title("All-Possible Pairs Diversity")
    plt.show()


def elitism_run(size=40, mut=0.2, gen=15):
    # Function just organizes the aforementioned functions and establishes parameter values including
    # population size, cull rate, mutation rate, and number of generations to be performed.
    pop_size = size
    cull_rate = 0.5
    mut_rate = mut
    generations = gen
    pop = population_generator(pop_size)
    longitudinal_fitness = list()
    longitudinal_avg_fitness = list()
    all_pp_div_list = list()
    for gen in range(generations):
        all_pp_div = all_possible_pairs(pop)
        all_pp_div_list.append(all_pp_div)
        fit, pop_fitness, best_fitness = fitness_calc(pop)
        print("\n**NEW GENERATION**\nGeneration:", gen + 1, "\nTop Fitness Value:", fit[0][0],
              "\nPhenotype:", fit[0][1], "\nPopulation Size:", len(fit), "\n")
        items_printer(fit[0][1])
        longitudinal_avg_fitness.append(pop_fitness)
        longitudinal_fitness.append(best_fitness)
        selected = selection(fit, cull_rate)
        new_pop = elitism_crossover(selected, pop_size)
        mut_pop = elitism_mutation(new_pop, mut_rate)
        pop = mut_pop
    plt_fitness(longitudinal_fitness, longitudinal_avg_fitness, generations, pop_size)
    plt_all_pp_diversity(all_pp_div_list, longitudinal_fitness)
    print("\n**Algorithm run with elitism**")


def new_pop_run(size=40, mut=0.2, gen=15):
    # Function just organizes the aforementioned functions and establishes parameter values including
    # population size, cull rate, mutation rate, and number of generations to be performed.
    pop_size = size
    cull_rate = 0.5
    mut_rate = mut
    generations = gen
    pop = population_generator(pop_size)
    longitudinal_fitness = list()
    longitudinal_avg_fitness = list()
    all_pp_div_list = list()
    for gen in range(generations):
        all_pp_div = all_possible_pairs(pop)
        all_pp_div_list.append(all_pp_div)
        fit, pop_fitness, best_fitness = fitness_calc(pop)
        print("\n**NEW GENERATION**\nGeneration:", gen + 1, "\nTop Fitness Value:", fit[0][0],
              "\nPhenotype:", fit[0][1], "\nPopulation Size:", len(fit), "\n")
        items_printer(fit[0][1])
        longitudinal_avg_fitness.append(pop_fitness)
        longitudinal_fitness.append(best_fitness)
        selected = selection(fit, cull_rate)
        new_pop = elitism_crossover(selected, pop_size)
        mut_pop = elitism_mutation(new_pop, mut_rate)
        pop = mut_pop
    plt_fitness(longitudinal_fitness, longitudinal_avg_fitness, generations, pop_size)
    plt_all_pp_diversity(all_pp_div_list, longitudinal_fitness)
    print("\n**Algorithm run with generational replacement**")


def main():
    user_in = input("Enter 1 to run the algorithm with generational replacement, "
                    "Enter 2 to run with elitism,"
                    " Otherwise hit enter for default parameters: ")
    if user_in == '1':
        size = int(input("Enter population size: "))
        mut = float(input("Enter mutation rate: "))
        gen = int(input("Enter number of generations: "))
        if size >= 2**12:
            raise Exception("Size cannot exceed 4096")
        if mut < 0 or mut > 1:
            raise Exception("0 < mutation rate < 1 condition cannot be defied")
        if size % 2 != 0 or size < 6:
            raise Exception("Population size should be an even number that is larger than 6")
        new_pop_run(size, mut, gen)
    elif user_in == '2':
        size = int(input("Enter population size: "))
        mut = float(input("Enter mutation rate: "))
        gen = int(input("Enter number of generations: "))
        if size >= 2 ** 12:
            raise Exception("Size cannot exceed 4096")
        if mut < 0 or mut > 1:
            raise Exception("0 < mutation rate < 1 condition cannot be defied")
        if size % 2 != 0 or size < 6:
            raise Exception("Population size should be an even number that is larger than 6")
        elitism_run(size, mut, gen)
    else:
        elitism_run()


main()
