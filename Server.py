from flask import Flask
from flask import render_template
from flask import request, send_file
from werkzeug.utils import secure_filename
from ApproximatorThread import ApproximatorThread
import SharedApproximator as sa
import Configuration as conf
import Types

# ----------------------
# --   Flask Setup    --
# ----------------------

app = Flask(__name__)


# ----------------------
# --      Routes      --
# ----------------------

@app.route('/', methods=['GET'])
def main():
    """
    Application settings
    """

    return render_template('set_config.html')

@app.post('/run')
def run():
    """
    Receive algorithm settings, store the input image on the file system
    and start the Approximator on another thread so that the main thread
    (Flask's) does not get blocked. I/O between frontend and backend will
    happen asynchronously.
    """

    input_image = request.files['input_image']
    if not input_image:
        return '<p>Error: Input Image is not valid</p>'
    input_image.save(f'{conf.UPLOAD_PATH}{secure_filename(input_image.filename)}')

    approximator_thread = ApproximatorThread(
        f'{conf.UPLOAD_PATH}{secure_filename(input_image.filename)}',
        request.form['algorithm'],
        int(request.form['generations']),
        int(request.form['population_size']),
        float(request.form['mutation_probability']),
        float(request.form['crossover_probability']),
        float(request.form['indpb']),
        int(request.form['polygons']),
        tournament_selection=request.form['tournament_selection'],
        tournament_size=int(request.form['tournament_size']),
        mu=int(request.form['mu']),
        lambda_=int(request.form['lambda']),
        sort_points='sort_points' in request.form,
        seed=int(request.form['seed']),
        save_evolving_image='save_images' in request.form,
        save_every=int(request.form['every']),
        output_folder=request.form['output_folder']
    )
    approximator_thread.start()

    return render_template('approximator.html',
                            algorithm=request.form['algorithm'],
                            generations=request.form['generations'],
                            population_size=request.form['population_size'],
                            mutation_probability=request.form['mutation_probability'],
                            crossover_probability=request.form['crossover_probability'],
                            indpb=request.form['indpb'],
                            polygons=request.form['polygons'],
                            tournament_selection=request.form['tournament_selection'],
                            tournament_size=request.form['tournament_size'],
                            mu=request.form['mu'],
                            lambda_=request.form['lambda'],
                            sort_points='sort_points' in request.form,
                            seed=request.form['seed'],
                            save_evolving_image='save_images' in request.form,
                            save_every=request.form['every'],
                            output_folder=request.form['output_folder']
                            )

@app.route('/progress')
def progress():
    """
    REST-like API for the frontend to know computation progress of
    the algorithm.
    It returns a JSON object containing:
    - how many generations have already been calculated
    - how many generations there are in total (that is, the computation target)
    - the most recent image, if any
    """

    result = {}

    result['completed'] = len(sa.approximator.logbook)
    result['total'] = sa.approximator.generations
    if len(sa.approximator.result_images) > 0:
        result['image'] = sa.approximator.result_images[-1]

    return result

@app.route('/image/<image_name>')
def get_image(image_name):
    """
    Sends the requested image from the file system
    """

    return send_file(f'{conf.IMAGE_PATH}{image_name}', mimetype='image/png')


# ----------------------
# --       Run        --
# ----------------------

if __name__ == '__main__':
    app.run()