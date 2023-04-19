import importlib
import configparser
import json
from json import JSONDecodeError
from pathlib import Path
import os

from train import parse_args, train

LOCAL_HPARAMS = "parameters.json"

def load_params():
    param_path = get_params_path()

    with open(param_path, "r") as f:
        return parse_nested_json(json.load(f))


def get_params_path():
    prefix = Path("/opt/ml/")
    param_path = prefix / "input/config/hyperparameters.json"

    if not param_path.exists():
        param_path = prefix / LOCAL_HPARAMS

    return param_path


def parse_nested_json(obj):
    """Parse nested string json
    Required because sagemaker formats nested json as string values
    """
    if isinstance(obj, dict):
        return {k: parse_nested_json(v) for k, v in obj.items()}
    if isinstance(obj, str):
        try:
            return parse_nested_json(json.loads(obj.replace("'", '"')))
        except JSONDecodeError:
            return obj
    return obj



def trainer():
    params = load_params()
    hyperparameters = params["hyperparameters"]

    args = parse_args()

    print("params", hyperparameters)
    for key, value in hyperparameters.items():
        if hasattr(args, key):
            setattr(args, key, value)

    print(args)
    print("text: ", args.text)
    print("O: ", args.O)
    print("iters: ", args.iters)

    train(args)

    