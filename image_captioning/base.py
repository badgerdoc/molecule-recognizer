from abc import ABC, abstractmethod
from typing import Tuple, Type, TypeVar, Dict, Union, Optional
import torch.nn as nn

from pydantic import BaseModel


ModelType = TypeVar('ModelType', bound=Type[Union['ConfigurableModel', nn.Module]])
ModelConfigType = TypeVar('ModelConfigType', bound=Type['MLModelBaseConfig'])


class ConfigurableModel:
    """A base class for models that can be created from config."""
    name: Optional[str] = None

    @classmethod
    @abstractmethod
    def from_config(cls, config: 'MLModelBaseConfig'):
        return


class LibRegistry:
    """
    A singleton that stores mappings:
    For models: name: str -> cls: Type
    For configs: cls_name: str -> cls: Type
    """
    class __Registry:
        def __init__(self):
            self.models: Dict[str, ModelType] = {}
            self.configs: Dict[str, ModelConfigType] = {}

    _instance: Optional[__Registry] = None

    def __init__(self):
        if LibRegistry._instance is None:
            LibRegistry._instance = LibRegistry.__Registry()

    def get_model(self, name: str) -> ModelType:
        cls = self._instance.models.get(name)
        if not cls:
            raise KeyError(f'No model registered under the name "{name}" in LibRegistry.')
        return cls

    def add_model(self, name: str, cls: ModelType):
        if self._instance.models.get(name):
            raise KeyError(
                f'Trying to register multiple models with name "{name}".'
            )
        elif not issubclass(cls, ConfigurableModel):
            raise TypeError(
                f'Can not register "{name}" because {cls.__name__}'
                f'is not subclass of {ConfigurableModel.__name__}.'
            )
        self._instance.models[name] = cls

    def get_config(self, name: str) -> ModelConfigType:
        cls = self._instance.configs.get(name)
        if not cls:
            raise KeyError(f'No config registered under the name "{name}" in LibRegistry.')
        return cls

    def add_config(self, name: str, cls: ModelConfigType):
        if self._instance.configs.get(name):
            raise KeyError(
                f'Trying to register multiple configs with name "{name}".'
            )
        self._instance.configs[name] = cls

    def __init_subclass__(cls, **kwargs):
        raise TypeError('LibRegistry can not be subclassed.')


def _register(cls: ModelType, name: str):
    cls.name = name
    LibRegistry().add_model(name, cls)
    return cls


def register_model(name: str):
    """Adds model to `LibRegistry` with associated name."""
    def wrap(cls: Type[nn.Module]):
        return _register(cls, name)
    return wrap


def get_model(config: 'MLModelBaseConfig') -> ModelType:
    """Construct model from its config."""
    name = config.name
    cls = LibRegistry().get_model(name)
    if not cls:
        raise KeyError(
            f'Could not find a class associated with name "{name}". If there is a '
            f'class with decorator `@register_model(name="{name}") import it in `image_captioning.__init__.py'
        )
    elif not issubclass(cls, ConfigurableModel):
        raise TypeError(f'{cls.__name__} should be subclass of {ConfigurableModel.__name__}.')
    return cls.from_config(config)


class RegisteredConfigMixin:
    """Registers config subclasses in LibRegistry."""

    def __init_subclass__(cls, **kwargs):
        cls_name = cls.__name__
        LibRegistry().add_config(cls_name, cls)


class MLModelBaseConfig(BaseModel, ABC, RegisteredConfigMixin):
    name: str

    @property
    @abstractmethod
    def id(self) -> str:
        """
        A string consisting of model's name and most important parameters, used to identify
        the model in checkpoints folder.
        """
        return


class EncoderBaseConfig(MLModelBaseConfig):
    size: Tuple[int, int]  # height, width

    @property
    def id(self) -> str:
        return f'{self.name}_{self.size[0]}x{self.size[1]}'


class DecoderBaseConfig(MLModelBaseConfig):
    @property
    def id(self) -> str:
        # TODO: consider adding some other parameters overriding this in subclasses
        return self.name
