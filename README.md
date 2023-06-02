# Introduction

Bio-Inspired Artificial Intelligence Project @ University of Trento, Spring 2023.

This project aims at reconstructing images through evolutionary algorithms (EAs) with limited resources (_e.g._ only 100 polygons).

The algorithm produces vectorized images that are not only visually appealing, but also capture the essence of the original image with a limited number of polygons.

The algorithm is not intended to create a perfect reconstruction of the given image, but rather to explore the artistic possibilities of polygon-based representation.


# How to install

```bash
pip install -r requirements.txt
```


# How to run

Move to the root directory of the project and run `Server.py`, for instance:

```
cd bio-inspired-ai-project
python Server.py
```

It will start a Web server using Flask. Note that the Web server is *not* intended for production use but only for local testing/development (more on Flask documentation).

On the command line, the URL to connect to will be shown. Visit that URL with your Web browser. A user interface will appear. You can customize the algorithm's parameters and then start the evolutionary process.