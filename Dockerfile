FROM ubuntu:16.04

# install ubuntu related libraries
RUN apt-get update && \
    apt-get -y install build-essential \
    python-dev python3-dev python3-pip wget curl vim

# set path to scripts

RUN mkdir -p /fashion/src/models/
RUN mkdir -p /fashion/src/processing/
RUN mkdir -p /fashion/models/
RUN mkdir -p /fashion/data/img/
RUN mkdir -p /fashion/data/Anno/
RUN mkdir -p /fashion/data/search_img/

COPY . /fashion/
RUN chmod +x **/*.py

# install required packages
RUN wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py && \
    pip3 install cycler==0.10.0 \
    h5py==2.9.0 \
    Keras==2.2.4 \
    Keras-Applications==1.0.7 \
    Keras-Preprocessing==1.0.9 \
    kiwisolver==1.1.0 \
    matplotlib==3.0.3 \
    numpy==1.16.3 \
    pandas==0.24.2 \
    Pillow==6.0.0 \
    pypandoc==1.4 \
    pyparsing==2.4.0 \
    python-dateutil==2.8.0 \
    pytz==2019.1 \ 
    PyYAML==5.1 \
    scikit-learn==0.21.2 \
    scipy==1.2.1 \
    six==1.12.0 \
    tensorboard==1.13.1 \ 
    tensorflow==1.13.1 \ 
    tensorflow-estimator==1.13.0 

RUN rm -rf /var/lib/apt/lists/*
RUN rm -rf /root/.cache

COPY /src/models/keras.json  ~/.keras/keras.json

# Set environment variables and working directory
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"

ENV PATH="/opt/program:${PATH}"

WORKDIR /opt/program

