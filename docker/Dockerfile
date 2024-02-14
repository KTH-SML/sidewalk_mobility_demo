# Example Dockerfile for SVEA images
#
# Author: Kaj Munhoz Arfvidsson

ARG BUILD_TAG
ARG ROSDISTRO

FROM $BUILD_TAG

ARG ROSDISTRO
ARG DEBIAN_FRONTEND=noninteractive

## Install dependencies using apt-get

RUN apt-get update -y

# # You can add apt dependencies like this...
# RUN apt-get install --no-install-recommends -y \
#         git curl

# This is for any added ROS packages
COPY src ./src
COPY requirements.txt ./
RUN rosdep install \
        --rosdistro $ROSDISTRO \
        --from-paths src \
        --ignore-src \
        -qry \
        && \
    pip install --ignore-installed -r requirements.txt && \ 
    pip install -U --force-reinstall "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Run catkin build on workspace (improves build speeds later on).
RUN catkin build

