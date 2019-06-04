import argparse
import functools
import json

from pylib import serialization


GLOBAL_COMMAND_PARSER = argparse.ArgumentParser()


def _serialization_wrapper(func):
    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        to_json = kwargs.pop("to_json", None)
        to_yaml = kwargs.pop("to_yaml", None)
        namespace = func(*args, **kwargs)
        if to_json:
            args_to_json(to_json, namespace)
        if to_yaml:
            args_to_yaml(to_yaml, namespace)
        return namespace
    return _wrapper


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected!')


def argument(*args, **kwargs):
    """Обертка для argparse.add_argument."""
    if 'type'in kwargs:
        if issubclass(kwargs['type'], bool):
            kwargs['type'] = str2bool
        elif issubclass(kwargs['type'], dict):
            kwargs['type'] = json.loads
    return GLOBAL_COMMAND_PARSER.add_argument(*args, **kwargs)


arg = argument


@_serialization_wrapper
def args(args=None, namespace=None):
    """Парсинг аргументов"""
    namespace = GLOBAL_COMMAND_PARSER.parse_args(args=args, namespace=namespace)
    return namespace


def args_to_json(path, namespace, **kwagrs):
    serialization.save_json(path, vars(namespace), **kwagrs)


def args_to_yaml(path, namespace, **kwagrs):
    serialization.save_yaml(path, vars(namespace), **kwagrs)
