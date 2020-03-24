# NGS (ns3 + OpenAI Gym + SUMO)

Welcome to NGS, an augmented version of the repository originally developed by [tkn-tub](https://github.com/tkn-tub/ns3-gym). 

In this augmented version we have managed to additionally integrate the SUMO traffic simulator. Hence all SUMO traffic scenarios are also supported.

Additionally, we performed all of our reinforcement learning using the agents present in the [stable-baselines](https://github.com/hill-a/stable-baselines) repository. Which is a repository with high-quality implementations of well known Deep RL algorithms.

Continuing to scroll down, you will find the three different subsections that will explains the required dependencies for:
* SUMO
* Stable Baselines
* ns3-gym

**Note**: You can go through the steps below for a more hands on installation process or you can run the installation helper script by entering the following command in the terminal:
```bash
./install_dep
```

The above command must be run the working directory of ns3-gym.

## Run simple traffic simulation
```bash
# The following two commands are needed only
# if this is the first time running
# ------------------------------------------ 
./waf configure --enable-examples
./waf build
# ------------------------------------------
# 1 = circle traffic scenario
# 2 = two lane highway merge scenario
# 3 = square traffic scenario
./launch_sumo scenario=[ 1 | 2 | 3 ]
```

## Train a traffic agent
You must have two terminal windows open to perform the following:

Terminal 1 (ns3 side): 
```bash
# This should be run the main working directory
./launch_sumo scenario=[ 1 | 2 | 3 ]
```

Terminal 2 (gym side):
```bash
# This should be run under rl_fyp/gym_rsu
python3 script.py train online [ alg_name ] scenario=[name] [ policy_kwargs ]
```

# SUMO

## What is SUMO

["Simulation of Urban MObility" (SUMO)](https://sumo.dlr.de/) is an open source,
highly portable, microscopic traffic simulation package designed to handle
large road networks and different modes of transport.

It is mainly developed by employees of the [Institute of Transportation Systems
at the German Aerospace Center](https://www.dlr.de/ts).


## Where to get it

You can download SUMO via our [downloads site](https://sumo.dlr.de/docs/Downloads.html).

As the program is still under development and is extended continuously, we advice you to
use the latest sources from our GitHub repository. Using a command line client
the following command should work:

        git clone --recursive https://github.com/eclipse/sumo

# Stable Baselines

Stable Baselines is a set of improved implementations of reinforcement learning algorithms based on OpenAI [Baselines](https://github.com/openai/baselines/).

## Documentation

Documentation is available online: [https://stable-baselines.readthedocs.io/](https://stable-baselines.readthedocs.io/)

## RL Baselines Zoo: A Collection of 100+ Trained RL Agents

[RL Baselines Zoo](https://github.com/araffin/rl-baselines-zoo). is a collection of pre-trained Reinforcement Learning agents using Stable-Baselines.

## Installation

**Note:** Stabe-Baselines supports Tensorflow versions from 1.8.0 to 1.14.0. Support for Tensorflow 2 API is planned.

### Prerequisites
Baselines requires python3 (>=3.5) with the development headers. You'll also need system packages CMake, OpenMPI and zlib. Those can be installed as follows

#### Ubuntu

```bash
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
```

#### Mac OS X
Installation of system packages on Mac requires [Homebrew](https://brew.sh). With Homebrew installed, run the following:
```bash
brew install cmake openmpi
```

#### Windows 10

To install stable-baselines on Windows, please look at the [documentation](https://stable-baselines.readthedocs.io/en/master/guide/install.html#prerequisites).

### Install using pip
Install the Stable Baselines package:
```
pip install stable-baselines[mpi]
```

This includes an optional dependency on MPI, enabling algorithms DDPG, GAIL, PPO1 and TRPO. If you do not need these algorithms, you can install without MPI:
```
pip install stable-baselines
```

Please read the [documentation](https://stable-baselines.readthedocs.io/) for more details and alternatives (from source, using docker).


ns3-gym
============

[OpenAI Gym](https://gym.openai.com/) is a toolkit for reinforcement learning (RL) widely used in research. The network simulator [ns–3](https://www.nsnam.org/) is the de-facto standard for academic and industry studies in the areas of networking protocols and communication technologies. ns3-gym is a framework that integrates both OpenAI Gym and ns-3 in order to encourage usage of RL in networking research.

Installation
============

1. Install all required dependencies required by ns-3.
```
# minimal requirements for C++:
apt-get install gcc g++ python

see https://www.nsnam.org/wiki/Installation
```
2. Install ZMQ and Protocol Buffers libs:
```
# to install protobuf-3.6 on ubuntu 16.04:
sudo add-apt-repository ppa:maarten-fonville/protobuf
sudo apt-get update

apt-get install libzmq5 libzmq5-dev
apt-get install libprotobuf-dev
apt-get install protobuf-compiler
```
3. Configure and build ns-3 project (if you are going to use Python virtual environment, please execute these commands inside it):
```
# Opengym Protocol Buffer messages (C++ and Python) are build during configure
./waf configure
./waf build
```

4. Install ns3gym located in src/opengym/model/ns3gym (Python3 required)
```
pip3 install ./src/opengym/model/ns3gym
```

5. (Optional) Install all libraries required by your agent (like tensorflow, keras, etc.).

6. Run example:
```
cd ./scratch/opengym
./simple_test.py
```

7. (Optional) Start ns-3 simulation script and Gym agent separately in two terminals (useful for debugging):
```
# Terminal 1
./waf --run "opengym"

# Terminal 2
cd ./scratch/opengym
./test.py --start=0
```

Contact
============
* American University of Beirut:
    * **Rayyan Nasr**: rrn13@mail.aub.edu
    * **Jihad Eddine Al Khurfan**: jia07@mail.aub.edu
    * **Ahmad Abou Adla**: aka38@mail.aub.edu

How to reference the original ns3-gym?
============

Please use the following bibtex :

```
@inproceedings{ns3gym,
  Title = {{ns-3 meets OpenAI Gym: The Playground for Machine Learning in Networking Research}},
  Author = {Gaw{\l}owicz, Piotr and Zubow, Anatolij},
  Booktitle = {{ACM International Conference on Modeling, Analysis and Simulation of Wireless and Mobile Systems (MSWiM)}},
  Year = {2019},
  Location = {Miami Beach, USA},
  Month = {November},
  Url = {http://www.tkn.tu-berlin.de/fileadmin/fg112/Papers/2019/gawlowicz19_mswim.pdf}
}
```
