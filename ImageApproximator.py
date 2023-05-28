from deap import tools
import numpy as np
import math

class ImageApproximator():
    """Functions to approximate an image with a set of polygons
    """

    def __init__(self, 
                 image_tools, 
                 mutation_probabilities=None,
                 sort_points=True
                ) -> None:
        """
        Construct an image approximator with the given parameters.
        """
        
        self.image_tools = image_tools
        self.sort_points = sort_points

        if mutation_probabilities:
            self.mutation_probabilities = mutation_probabilities
        else:
            self.mutation_probabilities = [
                0.5, # Mutating polygon's points,
                0.6, # Transform the polygon in a 4-vertex polygon
                0.8, # Update color and transparency
            ]

    def point_angles(self, x_coordinates, y_coordinates):
        """Compute the angles (in degrees) for a given list of coordinates with respect to their centroid

        The angles for each point (pair x, y) are computed with respect to the mean (also known
        as the 'centroid') of all the points provided.
        The angle is computed in degrees.

        Return a list of tuples, each of which contains the point as the first element and the polar
        coordinate (that is, the angle in degrees) of that point with respect to the centroid of all
        the points given.
        """

        x_mean = int(np.array(x_coordinates).mean())
        y_mean = int(np.array(y_coordinates).mean())

        points = []
        for x, y in zip(x_coordinates, y_coordinates):
            angle = math.atan2(y_mean - y, x_mean - x)
            polar_coordinate = math.degrees(angle)
            points.append((
                (x, y),
                polar_coordinate
            ))

        return points

    def points_polar_sort(self, x_coordinates, y_coordinates):
        """Sort the points given with respect to their polar coordinates (angles)
        """

        points = self.point_angles(x_coordinates, y_coordinates)
        
        # Sort points by angle (polar coordinate)
        points.sort(key=lambda p: p[1])

        return points
    
    def make_polygon(self, options=[(4, 0.6, False), (3, 0.2, True, (20, 150), (10, 30), (-20, 20)), (3, 0.2, False)]):
        """
        Generate a polygon with a given number of vertices (default at 3, for triangles).
        `vertices` represents the number of vertices with the associated probability.
        
        For instance, `vertices` = [(3, 0.5, True), (4, 0.5, False)] generates a triangle with probability
        0.5 and a 4-sided polygon with probability 0.5.
        Each element in the list is a tuple with the following structure: (#vertices, probability, move around)
        All the probabilities have to sum up to 1 and #vertices >= 3.
        Move around is used to generate points which only move around the first one generated in
        a constrained range, instead of generating them freely on the canvas.
        """

        if len(list(filter(lambda option: option[0] < 3, options))) > 0:
            raise ValueError("Minimum number of vertices for a polygon is 3")
        
        # Generate red, green, blue and alpha channels for the polygon
        # red, green, blue = np.random.choice(range(0, 235), 3)
        # alpha = np.random.randint(30, 90)
        red, green, blue = np.random.choice(range(0, 256), 3)
        alpha = np.random.randint(25, 100)

        choice = np.random.random()

        accumulated_probability = 0
        for option in options:
            accumulated_probability += option[1]

            # Probability not enough for this option,
            # go ahead and try next option
            if choice >= accumulated_probability:
                continue

            # Generate a random point within a given range and then generate other
            # points (the number on the input provided) as displacements from that
            # initial point
            if option[2]:
                starting_point = np.random.choice(range(option[3][0], option[3][1]), 2)
                coordinates = [starting_point[0], starting_point[1]]
                for i in range(4, len(option)):
                    displacement_range = option[i]
                    displacements = np.random.choice(range(displacement_range[0], displacement_range[1]), 2)
                    coordinates.append(displacements[0] + starting_point[0])
                    coordinates.append(displacements[1] + starting_point[1])

            # Generate random points, where the number depends on the input provided
            else:
                # coordinates = np.random.choice(range(10, 190), option[0] * 2)
                # Size is (width, height)
                coordinates_x = np.random.choice(range(0, self.image_tools.target_image.size[0]), option[0])
                coordinates_y = np.random.choice(range(0, self.image_tools.target_image.size[1]), option[0])
                coordinates = [coordinate for point in zip(coordinates_x, coordinates_y) for coordinate in point]

            if self.sort_points:
                points_sorted = self.points_polar_sort(coordinates[::2], coordinates[1::2])
            else:
                points_sorted = self.points_angles(coordinates[::2], coordinates[1::2])

            # Each point is a tuple (coordinates, polar coordinates), therefore
            # to access only (x, y) coordinates p[index][0] is necessary
            polygon = [(red, green, blue, alpha)]
            for p in points_sorted:
                polygon.append(p[0])

            return polygon
        
    def mutation_strategy(self, solution, indpb):
        """Mutation strategy for the evolutionary algorithm
        """

        # Choose a random polygon from the solution
        polygon = solution[np.random.randint(0, len(solution))]

        action = np.random.random()

        # Mutate the polygon's points
        if action < self.mutation_probabilities[0]:
            # Iterating over the points (x_i, y_i) in the polygon
            # Iterating over the coordinates of each point, that is
            # going through x_i and y_i
            coordinates = [coord for point in polygon[1:] for coord in point]

            # Apply a Gaussian mutation with mean 0 and standard deviation 10
            tools.mutGaussian(coordinates, 0, 10, indpb)

            # Assuming coordinates contains coordinates in the order x_i, y_i:
            # coordinates[::2] takes all the coordinates in even positions, that
            # is all x's.
            # coordinates[1::2] takes all the coordinates in odd position, that
            # is all y's
            x_coordinates = coordinates[::2]
            y_coordinates = coordinates[1::2]

            # Convert each coordinate (which may be either some x_i or y_i)
            # into an integer, since Gaussian mutation has transformed it into
            # a float number. Then, take the minimum between it and the size of
            # the image, so as to ensure the coordinate is always within the boundaries.
            # Eventually, take the maximum between that and 0, so as to ensure that,
            # in the end, 0 <= int(coord) <= image_size
            # x coordinates
            x_coordinates = [max(0, min(int(coord),self.image_tools.target_image.size[0])) for coord in x_coordinates]
            # y coordinates
            y_coordinates = [max(0, min(int(coord), self.image_tools.target_image.size[1])) for coord in y_coordinates]

            # Coordinates are then zipped so as to create (x_i, y_i)
            # tuples and eventually converted into a list
            polygon[1:] = list(zip(x_coordinates, y_coordinates))

        # Transform the polygon in a 4-point polygon. If it has less, one more is added,
        # if it has more, some are removed and one new is added, to get a total of four
        elif action < self.mutation_probabilities[1]:
            # Generate a new point to add to the polygon
            new_x = np.random.randint(0, self.image_tools.target_image.size[0])
            new_y = np.random.randint(0, self.image_tools.target_image.size[1])

            index = solution.index(polygon)
            
            if self.sort_points:
                points = self.points_polar_sort(
                    [polygon[1][0], polygon[2][0], polygon[3][0], new_x],
                    [polygon[1][1], polygon[2][1], polygon[3][1], new_y],
                )
            else:
                points = self.points_angles(
                    [polygon[1][0], polygon[2][0], polygon[3][0], new_x],
                    [polygon[1][1], polygon[2][1], polygon[3][1], new_y],
                )
            
            # Update the polygon polygon
            polygon = [
                (polygon[0][0], polygon[0][1], polygon[0][2], polygon[0][3]),
                (points[0][0]), (points[1][0]), (points[2][0]), (points[3][0]),
            ]
            
            # Replace the old polygon with the new, updated one
            solution[index] = polygon
            
        # Update color and transparency
        elif action < self.mutation_probabilities[2]:
            red, green, blue = np.random.choice(range(-20, 20), 3)
            alpha = np.random.randint(-20, 20)
            color_channels = polygon[0]
            polygon[0] = (
                max(0, min(255, color_channels[0] + red)),   # update red channel
                max(0, min(255, color_channels[1] + green)), # update green channel
                max(0, min(255, color_channels[2] + blue)),  # update blue channel
                max(0, min(255, color_channels[3] + alpha))  # update alpha channel
            )

        # Add a new polygon to the solution or, if there are more than 100,
        # shuffle them
        else:
            if len(solution) < 25:
                solution.append(self.make_polygon(options=[
                    (4, 0.8, False),
                    (3, 0.1, True, (20, 150), (10, 30), (-20, 20)),
                    (3, 0.1, False),
                ]))
            elif len(solution) < 50:
                solution.append(self.make_polygon(options=[
                    (4, 0.4, False),
                    (3, 0.6, False),
                ]))
            elif len(solution) < 75:
                solution.append(self.make_polygon(options=[
                    (4, 0.2, False),
                    (3, 0.7, True, (20, 150), (10, 30), (-20, 20)),
                    (3, 0.1, False),
                ]))
            elif len(solution) < 100:
                solution.append(self.make_polygon(options=[
                    (4, 0.2, False),
                    (3, 0.5, True, (20, 150), (10, 30), (-20, 20)),
                    (3, 0.3, False),
                ]))
            else:
                tools.mutShuffleIndexes(solution, indpb)

        return (solution, )