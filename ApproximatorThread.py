from deap import base
from multiprocessing.pool import Pool as Pool
from Approximator import Approximator
import threading
import SharedApproximator as sa

class ApproximatorThread(threading.Thread):
    """Wraps an `Approximator` in a separate thread so that it runs in a non-blocking way
    """

    def __init__(self,
                 image_path,
                 algorithm,
                 generations, population_size,
                 mutation_probability, crossover_probability,
                 indpb,
                 polygons,
                 mating_strategy='cxTwoPoint',
                 tournament_selection='SelTournament', tournament_size=None,
                 mu=None, lambda_=None,
                 sort_points=True,
                 seed=None,
                 save_evolving_image=False, save_every=None, output_folder=None,
                 resize_input_image_width=200
                ):
        """
        Construct an ApproximatorThread and pass all the parameters to an `Approximator`,
        which is constructed when the thread is run.
        """
        
        self.image_path = image_path

        self.algorithm = algorithm
        self.generations = generations
        self.population_size = population_size
        self.mutation_probability = mutation_probability
        self.crossover_probability = crossover_probability
        self.indpb = indpb
        self.polygons = polygons
        self.mating_strategy = mating_strategy
        self.tournament_selection = tournament_selection
        self.tournament_size = tournament_size
        self.mu = mu
        self.lambda_ = lambda_
        self.sort_points = sort_points
        self.seed = seed

        self.save_evolving_image = save_evolving_image
        self.save_every = save_every
        self.output_folder = output_folder

        self.resize_input_image_width = resize_input_image_width

        if save_evolving_image and (not save_every or not output_folder):
            raise ValueError('Every how many generations to save and output folder have to be given')

        super().__init__()

    def run(self):
        """Use `start()` to run on an actual separate thread.
        """

        toolbox = base.Toolbox()

        # Create a pool of processes
        # Processes are use instead of threads as the
        # program is computation-intensive, whereas there
        # is little I/O to be done
        process_pool = Pool()

        # Instantiate the approximator with all the parameters and run it
        sa.approximator = Approximator(
            self.image_path,
            self.algorithm,
            process_pool, toolbox,
            self.generations, self.population_size,
            self.mutation_probability, self.crossover_probability, self.indpb,
            self.polygons,
            self.mating_strategy,
            tournament_selection=self.tournament_selection,
            tournament_size=self.tournament_size,
            mu=self.mu, lambda_=self.lambda_,
            sort_points=self.sort_points,
            seed=self.seed,
            resize_input_image_width=self.resize_input_image_width
        )
        sa.approximator.run(save_evolving_image=self.save_evolving_image, save_every=self.save_every, output_folder=self.output_folder)

        if process_pool:
            process_pool.close()