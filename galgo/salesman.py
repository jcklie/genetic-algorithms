from random import shuffle, random, seed

from util import *

data = [
    ("Kyiv",      (50.4500,  30.5000) ), # 50° 27′ N,  30° 30′ E
    ("Budapest",  (47.5000,  19.0500) ), # 47° 30′ N,  19°  3′ E
    ("Teheran",   (35.7000,  51.4167) ), # 35° 42′ N,  51° 25′ E
    ("Beijing",   (39.9000, 116.3833) ), # 39° 54′ N, 116° 23′ E
    ("Jerusalem", (31.7833,  35.2167) ), # 31° 47′ N,  35° 13′ E
    ("Bucharest", (44.4167,  26.1000) ), # 44°25′57″N 26°6′14″E
    ("Hamilton",  (43.2500,  79.8667) ), # 43°15′N 79°52′W
    ("Toronto",   (42.7000,  79.4000) ), # 43°42′N 79°24′W 
]

seed(0)

def create_random_trip(cities):
    trip = [city for city in lrange(cities)]
    shuffle(trip)
    return tuple(trip)

def initialize_population(data, population_size):
    return [create_random_trip(data) for unused in range(population_size)]

def get_coords(data, i):
    return data[i][1]

# Fitness
def evaluate_population(data, solution):
    assert len(solution) >= 2
    
    first = get_coords(data, solution[0])
    last = get_coords(data, solution[-1])
    score = distance(first, last)
    
    for i in range( len(solution) - 1 ):
        start = get_coords(data, solution[i])
        end  = get_coords(data, solution[i+1])
        score += distance(start, end)
    return score

def terminated():
    pass

def select(population, scores):
    """
    1)  The fitness function is evaluated for each individual, 
        providing fitness values, which are then normalized. 
        Normalization means dividing the fitness value of each 
        individual by the sum of all fitness values, so that 
        the sum of all resulting fitness values equals 1.
        
    2)  The population is sorted by descending fitness values.
    
    3)  Accumulated normalized fitness values are computed 
        (the accumulated fitness value of an individual is 
        the sum of its own fitness value plus the fitness 
        values of all the previous individuals). The accumulated 
        fitness of the last individual should be 1 (otherwise 
        something went wrong in the normalization step).
        
    4)  A random number R between 0 and 1 is chosen.
    
    5)  The selected individual is the first one whose 
        accumulated normalized value is greater than R.    
    """
    
    size = len(population)
    print 'scores', scores
    summed_score = sum(scores)
    
    normalized_score = [x / summed_score for x in scores]    
    normalized_score.sort()
    
    accumulated_score = []
    
    print size
    
    for i in range(size):
        accumulated_score.append( sum(normalized_score[:i + 1])  )
        
    R = random()
    
    print accumulated_score
    
    for s in accumulated_score:
        if R > s:
            return s
    else:
        print('Nothing has been gambled out!')
        return 

def crossover(mother, father):
    """ Order 1 Crossover is a fairly simple permutation crossover. 
    Basically, a swath of consecutive alleles from parent 1 drops 
    down, and remaining values are placed in the child in the 
    order which they appear in parent 2.     
    """    
    assert len(mother) == len(father)
    
    size = len(mother)    

    newborn = [-1 for i in range(size)]
    
    u_border = int(size / 4.0)
    o_border = int(3.0 / 4.0 * size)
        
    for i in range(u_border, o_border):
        newborn[i] = mother[i]
    
    remains = (x for x in father if x not in newborn)
    
    for i in range(size):
        if newborn[i] == -1:
            newborn[i] = next(remains)
    
    return tuple(newborn)   

def mutate():
    pass

def accept():
    pass

if __name__ == '__main__':
    seed(0)
    
    population_size = 10 
    
    # Generate random population of chromosomes
    population = initialize_population(data, population_size)
    
    # Evaluate the fitness of each chromosome in the population
    scores = [evaluate_population(data, solution) for solution in population]
    
    m = (3, 5, 7, 2, 1, 6, 4, 8)
    f = (2, 5, 7, 6, 8, 1, 3, 4)
    
    # Create, accept, and test a new population:
    while not terminated():
        # Select according to fitness
        mother, father = select(population, score)
            
        # With a crossover probability perform crossover or copy parents
        offspring = crossover(mother, father)
           
        # With a mutation probability mutate offspring at each position in chromosome
        mutated = mutate(offspring)
            
        if not accept(mutated):
            continue
            
        score =  evaluate_population(mutated)
