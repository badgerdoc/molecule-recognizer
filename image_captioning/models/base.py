from abc import ABC, abstractmethod
from typing import Tuple, Type, Dict, Union
import torch.nn as nn

from pydantic import BaseModel


class MLModelBaseConfig(BaseModel, ABC):
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


class ConfigurableModel:

    @classmethod
    @abstractmethod
    def from_config(cls, config: MLModelBaseConfig):
        return


class ModelRegistry:
    models: Dict[str, Type[Union[ConfigurableModel, nn.Module]]] = {}


def _register(cls: Type[nn.Module], name: str):
    if ModelRegistry.models.get(name):
        raise KeyError(
            f'Trying to assign same name "{name}" to multiple models with `register_model` decorator.'
        )
    elif not issubclass(cls, ConfigurableModel):
        raise TypeError(
            f'Can not register "{name}" because {cls.__name__}'
            f'is not subclass of {ConfigurableModel.__name__}.'
        )
    ModelRegistry.models[name] = cls
    cls.name = name
    return cls


def register_model(name: str):
    """Adds model to `ModelRegistry` with associated name."""
    def wrap(cls: Type[nn.Module]):
        return _register(cls, name)
    return wrap


def get_model(config: MLModelBaseConfig) -> Type[Union[ConfigurableModel, nn.Module]]:
    """
    Construct model from configuration file.
    """
    name = config.name
    cls = ModelRegistry.models.get(name)
    if not cls:
        raise KeyError(
            f'Could not find a class associated with name "{name}". If there is a '
            f'class with decorator `@register_model(name="{name}") import it in `image_captioning.__init__.py'
        )
    elif not issubclass(cls, ConfigurableModel):
        raise TypeError(f'{cls.__name__} should be subclass of {ConfigurableModel.__name__}.')
    return cls.from_config(config)
