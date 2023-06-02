import numpy as np
import random
from deap import tools, algorithms, creator
from ImageTools import ImageTools
from ImageApproximator import ImageApproximator
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

class Approximator():
    """Image approximator abstracting the interface of the evolutionary algorithm
    """

    def __init__(self, image_path,
                 algorithm, process_pool, toolbox,
                 generations, population_size,
                 mutation_probability, crossover_probability,
                 indpb,
                 polygons,
                 mating_strategy='cxTwoPoint',
                 tournament_selection='SelTournament', tournament_size=None,
                 mu=None, lambda_=None,
                 mutation_probabilities=None, sort_points=True,
                 seed=None,
                 resize_input_image_width=None
                ) -> None:
        """
        Construct an image approximator and provide an high-level interface
        to the underlying genetic algorithm.

        If `tournament_selection` is SelTournament, `tournament_size` has to be provided.
        If `algorithm` is `EaMuPlusLambda` or `EaMuCommaLambda`, both `mu` and `lambda_`
        have to be provided.
        """        

        self.image_tools = ImageTools(image_path)
        if resize_input_image_width:
            self.image_tools.resize_image(resize_input_image_width)

        self.image_approximator = ImageApproximator(self.image_tools, mutation_probabilities=mutation_probabilities, sort_points=sort_points)

        self.algorithm = algorithm
        self.process_pool = process_pool
        self.toolbox = toolbox
        self.generations = generations
        self.population_size = population_size
        self.mutation_probability = mutation_probability
        self.crossover_probability = crossover_probability
        self.indpb = indpb
        self.polygons = polygons

        if mating_strategy == 'cxTwoPoint':
            self.mating_strategy = tools.cxTwoPoint

        self.tournament_selection = tournament_selection
        self.tournament_size = tournament_size

        self.mu = mu
        self.lambda_ = lambda_

        if tournament_selection == 'SelTournament' and tournament_size is None:
            raise ValueError('Tournament size has to be specified for selection tournament!')
        
        if (algorithm == 'EaMuPlusLambda' or algorithm == 'EaMuCommaLambda') and (mu is None or lambda_ is None):
            raise ValueError(f'{algorithm} was chosen, but mu or lambda were not given')
        
        self.seed = seed
        self.result_images = []

    def run(self, save_evolving_image=False, save_final_plot=True, save_every=None, output_folder=None):
        """Run the evolutionary algorithm

        The evolutionary algorithm to be run is the one specified in the constructor.
        It is run with the given parameters.

        By setting the three parameters of `run()`, images can be saved every `save_every`
        generations on the file system at the given directory (`output_folder`). Saving every
        generation can be useful but it may take longer, as the process has to write on the disk
        much more often.
        `output_folder` has to be provided anyway, since it is used to store the final outcome
        computed by the algorithm.
        """
        
        # Set random seeds for reproducible generations
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Setup output evolving images
        if output_folder is None:
                raise ValueError('Output folder not given. It is necessary to save the final outcome')
        if save_evolving_image:
            if save_every is None:
                raise ValueError('Saving every n generations, but n is not given')
            if not output_folder.endswith('/'):
                output_folder = output_folder + '/'

        # ----------------------
        # -- Algorithm Setup  --
        # ----------------------

        # Go multiprocessing if a process pool was given to speed up computation
        if self.process_pool is not None:
            self.toolbox.register('map', self.process_pool.map)
            
        # Individuals factory
        self.toolbox.register('individual', tools.initRepeat, creator.Individual, self.image_approximator.make_polygon, n=self.polygons)
        
        # Population factory, which falls back onto individuals factory
        self.toolbox.register('population', tools.initRepeat, list, self.toolbox.individual)
        
        # Evaluation function, to assess individuals' fitness
        self.toolbox.register('evaluate', self.image_tools.evaluate)
        
        # Mutation strategy
        self.toolbox.register('mutate', self.image_approximator.mutation_strategy, indpb=self.indpb)

        # Mating strategy
        self.toolbox.register('mate', self.mating_strategy)

        if self.tournament_selection == 'DoubleTournament':
            self.toolbox.register('select', tools.selDoubleTournament, fitness_first=True, parsimony_size=1, fitness_size=9)
        elif self.tournament_selection == 'SelTournament':
            self.toolbox.register('select', tools.selTournament, tournsize=self.tournament_size)
        elif self.tournament_selection == 'SelRoulette':
            self.toolbox.register('select', tools.selRoulette)

            self.result_images = []

        population = self.toolbox.population(n=self.population_size)
        hall_of_fame = tools.HallOfFame(3)

        stats = tools.Statistics(lambda x: x.fitness.values[0])
        stats_size = tools.Statistics(key=len)
        mstats = tools.MultiStatistics(fitness=stats, size=stats_size)

        mstats.register('min', np.min)
        mstats.register('max', np.max)
        mstats.register('average', np.mean)
        mstats.register('median', np.median)
        mstats.register('std', np.std)


        self.logbook = tools.Logbook()
        self.logbook.header = ['gen'] + mstats.fields

        # ----------------------
        # - Solution Evolution -
        # ----------------------

        for generation in range(self.generations):
            if self.algorithm == 'EaMuPlusLambda':
                population, log = algorithms.eaMuPlusLambda(
                    population, self.toolbox,
                    mu=self.mu, lambda_=self.lambda_,
                    cxpb=self.crossover_probability, mutpb=self.mutation_probability, 
                    ngen=1, stats=mstats, halloffame=hall_of_fame, verbose=False)
            
            elif self.algorithm == 'EaMuCommaLambda':
                 population, log = algorithms.eaMuCommaLambda(
                    population, self.toolbox,
                    mu=self.mu, lambda_=self.lambda_,
                    cxpb=self.crossover_probability, mutpb=self.mutation_probability,
                    ngen=1, stats=mstats, halloffame=hall_of_fame, verbose=False)
                 
            elif self.algorithm == 'EaSimple':
                population, log = algorithms.eaSimple(
                    population, self.toolbox, 
                    cxpb=self.crossover_probability, mutpb=self.mutation_probability,
                    ngen=1, stats=mstats, halloffame=hall_of_fame, verbose=False)

            record = mstats.compile(population)
            self.logbook.record(gen=generation + 1, **record)
            print(self.logbook.stream)

            if save_evolving_image and generation % save_every == 0:
                image = self.image_tools.draw_solution(population[0])
                image.save(f'{output_folder}solution_generation_{generation}.png')
                self.result_images.append(f'solution_generation_{generation}.png')

        # ----------------------
        # -- End of evolution --
        # ----------------------

        if save_evolving_image:
            image = self.image_tools.draw_solution(population[0])
            image.save(f'{output_folder}solution_generation_{generation}.png')
            self.result_images.append(f'solution_generation_{generation}.png')

        image = self.image_tools.draw_solution(population[0])
        image.save(f'{output_folder}solution_generation_{self.generations}.png')
        self.result_images.append(f'solution_generation_{self.generations}.png')

        if save_final_plot:
            self.save_stats(output_folder)

    def save_stats(self, output_folder):
        """Generate summary plots and save them on the file system.
        """

        gen = self.logbook.select('gen')

        # Fitness information
        fit_mins = self.logbook.chapters['fitness'].select('min')
        fit_maxs = self.logbook.chapters['fitness'].select('max')
        fit_avgs = self.logbook.chapters['fitness'].select('average')

        # Individuals' size information
        size_mins = self.logbook.chapters['size'].select('min')
        size_maxs = self.logbook.chapters['size'].select('max')
        size_avgs = self.logbook.chapters['size'].select('average')

        # -- Fitness vs Size plot --
        fig, ax1 = plt.subplots()
        line1 = ax1.plot(gen, fit_avgs, "b-", label="Average Fitness")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Fitness", color="b")
        for tl in ax1.get_yticklabels():
            tl.set_color("b")

        ax2 = ax1.twinx()
        line2 = ax2.plot(gen, size_avgs, "r-", label="Average Size")
        ax2.set_ylabel("Size", color="r")
        for tl in ax2.get_yticklabels():
            tl.set_color("r")

        lns = line1 + line2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc="center right")

        plt.savefig(f'{output_folder}fitness_vs_size.png')


        # -- Overall fitness plot --
        fig, ax1 = plt.subplots()
        line1 = ax1.plot(gen, fit_mins, "r-", label="Minimum Fitness")
        line2 = ax1.plot(gen, fit_avgs, "b-", label="Average Fitness")
        line3 = ax1.plot(gen, fit_maxs, "g-", label="Maximum Fitness")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Fitness")

        lns = line1 + line2 + line3
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc="center right")

        plt.savefig(f'{output_folder}fitness_overall.png')


        # -- Overall size plot --
        fig, ax1 = plt.subplots()
        line1 = ax1.plot(gen, size_mins, "r-", label="Minimum Size")
        line2 = ax1.plot(gen, size_avgs, "b-", label="Average Size")
        line3 = ax1.plot(gen, size_maxs, "g-", label="Maximum Size")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Size (#polygons)")

        lns = line1 + line2 + line3
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc="center right")

        plt.savefig(f'{output_folder}size_overall.png')