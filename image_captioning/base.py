from abc import ABC, abstractmethod
from typing import Tuple, Type, Dict, Union
import torch.nn as nn

from pydantic import BaseModel


class ConfigurableModel:

    @classmethod
    @abstractmethod
    def from_config(cls, config: 'MLModelBaseConfig'):
        return


class LibRegistry:
    models: Dict[str, Type[Union[ConfigurableModel, nn.Module]]] = {}
    configs: Dict[str, Type['MLModelBaseConfig']] = {}

    def __new__(cls, *args, **kwargs):
        raise TypeError('LibRegistry can not be instantiated.')

    def __init_subclass__(cls, **kwargs):
        raise TypeError('LibRegistry can not be subclassed.')


def _register(cls: Type[nn.Module], name: str):
    if LibRegistry.models.get(name):
        raise KeyError(
            f'Trying to assign same name "{name}" to multiple models with `register_model` decorator.'
        )
    elif not issubclass(cls, ConfigurableModel):
        raise TypeError(
            f'Can not register "{name}" because {cls.__name__}'
            f'is not subclass of {ConfigurableModel.__name__}.'
        )
    LibRegistry.models[name] = cls
    cls.name = name
    return cls


def register_model(name: str):
    """Adds model to `LibRegistry` with associated name."""
    def wrap(cls: Type[nn.Module]):
        return _register(cls, name)
    return wrap


def get_model(config: 'MLModelBaseConfig') -> Type[Union[ConfigurableModel, nn.Module]]:
    """
    Construct model from configuration file.
    """
    name = config.name
    cls = LibRegistry.models.get(name)
    if not cls:
        raise KeyError(
            f'Could not find a class associated with name "{name}". If there is a '
            f'class with decorator `@register_model(name="{name}") import it in `image_captioning.__init__.py'
        )
    elif not issubclass(cls, ConfigurableModel):
        raise TypeError(f'{cls.__name__} should be subclass of {ConfigurableModel.__name__}.')
    return cls.from_config(config)


class RegisteredConfigMixin:

    def __init_subclass__(cls, **kwargs):
        cls_name = cls.__name__
        if not LibRegistry.configs.get(cls_name):
            LibRegistry.configs[cls_name] = cls
        else:
            raise KeyError(
                f'Config class "{cls_name}" already registered in `LibRegistry`.'
            )


class MLModelBaseConfig(BaseModel, ABC, RegisteredConfigMixin):
    name: str

    @property
    @abstractmethod
    def id(self) -> str:
        return


class EncoderBaseConfig(MLModelBaseConfig):
    size: Tuple[int, int]

    @property
    def id(self) -> str:
        return f'{self.name}_{self.size[0]}x{self.size[1]}'


class DecoderBaseConfig(MLModelBaseConfig):
    @property
    def id(self) -> str:
        # TODO: consider adding some other parameters overriding this in subclasses
        return self.name
