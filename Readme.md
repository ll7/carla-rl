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

0.9.10.1 requires `-opengl` during the server startup:

```shell
./CarlaUE4.sh -opengl
```


## check your python path

```shell
export PYTHONPATH=$PYTHONPATH:/home/luttkule/git/carla0.9.10.1-bin/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
```


