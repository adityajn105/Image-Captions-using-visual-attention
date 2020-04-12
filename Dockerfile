FROM continuumio/miniconda3:4.8.2

RUN conda install -q -y opencv=3.4.2 flask=1.0.3 tensorflow=2.1.0S

COPY app /home/app

COPY saved_objects /home/saved_objects


