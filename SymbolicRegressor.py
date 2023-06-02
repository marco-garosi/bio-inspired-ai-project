from ImageTools import ImageTools
import numpy as np
import random

import matplotlib.pyplot as plt

import operator
import math

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

import multiprocessing

"""
-------------------------------
|         Parameters          |
-------------------------------
"""

GP_POP_SIZE = 300               # [GP] Population size
GP_NGEN = 800                   # [GP] Number of generations
GP_CXPB, GP_MUTPB = 0.5, 0.5    # [GP] Crossover and Mutation probabilities
GP_TRNMT_SIZE = 4               # [GP] Tournament size
GP_HOF_SIZE = 2                 # [GP] Hall-of-Fame size
ALGORITHM = "Mu+Lambda"         # [EA] Choose between "Mu+Lambda" and "Simple"
seed = 42                       # [RNG] Seed

source_image = './input_images/square_lisa.png'
image_size = 32                 # [Img] Resizing dimension

"""
-------------------------------
|            Setup            |
-------------------------------
"""

image_tools = ImageTools(source_image)
image_tools.original_image = image_tools.original_image.convert('L')
image_tools.original_image = image_tools.original_image.resize((image_size, image_size))

width, height = image_tools.original_image.size
pixels = image_tools.original_image.load()


"""
-------------------------------
|          Operators          |
-------------------------------
"""

def protectedDiv(left, right):
    """
    Return left / right if right is not 0.
    Return 1 otherwise.
    """

    try:
        return left / right
    except ZeroDivisionError:
        return 1

def general_exp(base, exponent):
    """
    Return base^exponent
    """

    try:
        return math.pow(base, exponent)
    except:
        return 1

def protected_log(n):
    """
    Return log_e(n).
    """

    try:
        return math.log(n)
    except:
        return 1

# Making the primitive set

pset = gp.PrimitiveSetTyped("MAIN", [float], float)

pset.addPrimitive(operator.add, [float, float], float)
pset.addPrimitive(operator.sub, [float, float], float)
pset.addPrimitive(operator.mul, [float, float], float)
pset.addPrimitive(protectedDiv, [float, float], float)
pset.addPrimitive(operator.neg, [float], float)
pset.addPrimitive(math.cos, [float], float)
pset.addPrimitive(math.sin, [float], float)
pset.addPrimitive(protected_log, [float], float)

try:
    del creator.FitnessMinSR
    del creator.IndividualSR
except:
    pass
creator.create("FitnessMinSR", base.Fitness, weights=(-1.0,))
creator.create("IndividualSR", gp.PrimitiveTree, fitness=creator.FitnessMinSR)


"""
-------------------------------
|     Evaluation Function     |
-------------------------------
"""

def evalSymbReg(individual, points, target, compile):
    # Compile the tree expression into an actual Python function
    gpFunction = compile(expr=individual)

    # Evaluate the MSE (Mean Squared Error) between the expression
    # and the target values
    sqerrors = ((gpFunction(x) - target[x])**2 for idx, x in enumerate(points))
    
    return math.fsum(sqerrors) / len(points),


"""
-------------------------------
|    Evolutionary Process     |
-------------------------------
"""

def approximate_series(pixel_series, width, height, pool=None):
    """
    Approximate `pixel_series` through Genetic Programming.

    Parameters for GP are not passed to this function as they are
    global variables.
    """

    toolbox = base.Toolbox()
    if pool:
        toolbox.register("map", pool.map)
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.IndividualSR, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    # Evolutionary process setup
    toolbox.register("evaluate", evalSymbReg, points=[x for x in range(0, height)], target=pixel_series, compile=toolbox.compile)
    toolbox.register("select", tools.selTournament, tournsize=GP_TRNMT_SIZE)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    random.seed(seed)

    pop = toolbox.population(n=GP_POP_SIZE)
    hof = tools.HallOfFame(GP_HOF_SIZE)

    # Statistics
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    if ALGORITHM == 'Simple':
        final_pop,logbook=algorithms.eaSimple(pop, toolbox, GP_CXPB, GP_MUTPB, GP_NGEN, \
                                            stats=mstats, halloffame=hof, verbose=False)
    elif ALGORITHM == 'Mu+Lambda':
        final_pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, 30, 50, GP_CXPB, GP_MUTPB, GP_NGEN, \
                                    stats=mstats, halloffame=hof, verbose=False)
    else:
        raise ValueError(f'Supported algorithms: Mu+Lambda, Simple. {ALGORITHM} is not supported.')

    gen = logbook.select("gen")
    fit_mins = logbook.chapters["fitness"].select("min")
    size_avgs = logbook.chapters["size"].select("avg")

    # Get the best function at the end of the evolutionary process
    gpFunction = toolbox.compile(expr=hof[0])

    # Predict the outcomes (approximate pixel_series)
    pred = np.array([gpFunction(x) for x in range(height)])
    
    return (gpFunction, pred)

def process_series(pixel_series, col, width, height, results):
    """
    Process a series of pixels (a scanline).
    """

    print(f"-- Started: processing column {col}")
    gpFunction, result = approximate_series(pixel_series, width, height)
    print(f"-- Completed: processing column {col}")

    results[col] = result


"""
-------------------------------
|    Running the algorithm    |
-------------------------------
"""

if __name__ == '__main__':
    """
    A pool of processes is instantiated. Then, each process runs a distinct
    instance of an Evolutionary Algorithm on a scanline, thus processing
    scanlines in parallel.
    This usage is different than that of standard DEAP, which usually applies
    multiprocessing to evaluation functions: here the whole process is
    parallelized.
    """

    pool = multiprocessing.Pool()

    # Shared list
    manager = multiprocessing.Manager()
    results = manager.list([None] * width)

    # Start the jobs
    jobs = []
    for col in range(width):
        # Get the scanline and normalize it
        pixel_series = np.array([pixels[col, row] for row in range(height)], dtype='float64')
        pixel_series /= max(pixel_series)

        job = pool.apply_async(process_series, (pixel_series, col, width, height, results))
        jobs.append(job)

    for job in jobs:
        job.get()

    pool.close()
    pool.join()

    # All the jobs are completed, therefore all the results are
    # available. They can be converted into a plain Python list
    results = list(results)

    try:
        # And the plain Python list can be converted into a numpy array,
        # which is then transposed to make it look right
        reconstructed_image = np.array(results).T

        # Results are stored on the file system
        np.savetxt('./reconstructed.csv', reconstructed_image, delimiter=',')

        # The result is shown
        plt.imshow(reconstructed_image, cmap='gray')
        plt.show()
    except:
        print('Error in converting to numpy array')