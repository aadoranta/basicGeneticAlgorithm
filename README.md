# basicGeneticAlgorithm
A basic genetic algorithm developed from scratch in python.

Genetic Algorithm Project Description

1. The problem can be defined as a genetic algorithm in the following way:
 - The genotype will be all possible ways to fill the backpack. Each of the 12 items will be represented by a bit in a 12 bit binary
   binary number (1 if the item is in the backpack, 0 otherwise).
 - The selection of the initial population will be a random selection of n binary numbers between 0 and 2^12 - 1 where 0 represents no
   items being in the backpack and 2^12 - 1 represents all items being in the backpack. Every other binary number in between represents
   some other arrangement of items
 - Phenotypes of a population will be evaluated against a fitness function. The fitness function will return a value of 0 if the weight
   of the items is more than the max weight allowed (250), otherwise the fitness function will evaluate to the sum of the value of the
   items currently in the backpack. The goal is to maximize the fitness function.
 - The top 50% of phenotypes will be selected to reproduce based on the one-point crossover mutation algorithm.
 - One point mutation will then be applied. 
 - If elitism is applied, the top 2 individuals from the previous generation will join the new generation. Otherwise, the new
   generation will be entirely comprised of the mutated individuals.
 - This process will be repeat for a specified number of iterations.
2. The genotype for this problem is all arrangements of 1s and 0s in a binary number of bit-length = 12. This will effectively represent
   all possible arrangements of items in the backpack. Each bit corresponds to a single item (specified on the assignment sheet) and will
   be defined by a specific weight and importance value.
3. The fringe operations used in the project will be one-point crossover and one-point mutation. For reproduction, one-point crossover
   is employed meaning that the child of two parents will be composed of some amount of parent 1 and some amount of parent 2. The sum of
   the length of the components of parent 1 and parent 2 will be equal to the length of either parent; however, the length of each componenent 
   will be random. One point mutation means that a random position on a child (where the child is selected with some probability < 1 from
   the population) will be changed from one binary value to another (ex. 1 -> 0, 0 -> 1).
4. Every generation, only the top 50% of the population will be selected to reproduce, while the bottom 50% is discarded.

***ASSUMPTIONS***
 - implemented some interface options where you can choose to run the algorithm with or without elitism, determine the population size,
   determine the mutation rate, and determine the number of generations.
 - showed a plot demonstrating the average population fitness and best fitness vs. the number of generations - a fitness graph. I also 
   showed a graph of the all-diversity pairs and best fitness vs. number of generations. These graphs gave me some insight on how the 
   algorithm worked and I hope they're at least interesting to see.

***RUN INSTRUCTIONS***
I included two files - one that has the matplotlib plots enabled, and one with the matplotlib plot functions commented out. I could not
run the file with the plots in Idle, so if you're running it in that environment and just want to determine functionality, use 
genetic_algorithm_no_plt. In order to work, the codes simply need to be run. Prompts will show up which should be straightforward to follow
and results are printed as well as graphed.
