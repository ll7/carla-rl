# Carla Reinforcement Learning

## preperation

- Download Carla0.9.10.13
- use pip carla api


`conda env create -f environment.yml`

`conda activate carla-rl`

```shell
conda activate myenv
conda env update --file local.yml --prune
```

## carla

0.9.10.1 requires `-opengl` during the server startup due to [issue](https://github.com/carla-simulator/carla/issues/4328):

```shell
./CarlaUE4.sh -opengl
```

0.9.13 works fine

## check your python path

changing pythonpath requires restart?
set pythonpath in ~/.bashrc

```shell
export PYTHONPATH=$PYTHONPATH:/home/luttkule/git/carla0.9.10.1-bin/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
```

check pythonpath with

```shell
python -c "import carla; print(carla.__file__)"
```

## change map

https://github.com/carla-simulator/carla/issues/4901

CarlaUE4/Config/DefaultEngine.ini

not feasible to change default map easily

## conda

conda env create -f environment.yml

currently `conda activate carla-rl-py37-c13`

## IP address

provide the IP adress of the carla server in the ip-host.txt file if you want to run the server somewhere else and use the carla_env.read_IP_from_file() function


