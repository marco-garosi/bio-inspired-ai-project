from deap import base, creator

# ----------------------
# --   Global setup   --
# ----------------------

creator.create('FitnessMax', base.Fitness, weights=(1.0,))
# class FitnessMax(base.Fitness):
#     def __init__(self, values=()):
#         self.weights = (1.0, )
#         super().__init__(values)

creator.create('Individual', list, fitness=creator.FitnessMax)
# class Individual(list):
#     # fitness = FitnessMax()

#     # pass

#     def __init__(self):
#         self.fitness = FitnessMax()