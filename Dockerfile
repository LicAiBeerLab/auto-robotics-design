FROM ubuntu:22.04

RUN apt-get update 

# Install base utilities
RUN apt-get update && \
    apt-get install -y wget git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda


# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH
COPY environment_jmoves.yml .
RUN conda env create -f environment_jmoves.yml

# Install our package 
COPY . ./jmoves_env
RUN conda run -n j_moves pip install -e ./jmoves_env

