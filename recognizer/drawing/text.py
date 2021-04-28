import re
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

from typing import Optional, Union

from PIL import ImageFont, ImageDraw, Image

from recognizer.image_processing.utils import extract_content, save_img

DEFAULT_CANVAS_SIZE = (300, 300)

SPECIAL_SYMBOL = 'special_symbol'

ATOM_SYMBOL = 'atom_symbol'

ATOM_REGEX = r'([A-Z][a-z]?)'
SPECIAL_SYMBOL_REGEX = r'((_\[(\d)])|(\^\[([-|+|\d]{1,3})\]))+'


class TextRendererConfig:
    def __init__(
        self,
        font_path: Path,
        font_size: int = 18,
        subscript_size: int = 9,
        superscript_size: int = 9,
        main_spacing: int = 9,
        special_spacing: int = 5,
    ):
        self.main_spacing = main_spacing
        self.special_spacing = special_spacing
        self.superscript_size = superscript_size
        self.subscript_size = subscript_size
        self.font_size = font_size
        self.font_path = font_path


@dataclass
class SpecialSymbol:
    subscript: Optional[str]
    superscript: Optional[str]
    type_: str = field(default=SPECIAL_SYMBOL, init=False)

    @property
    def width(self):
        if not self.superscript:
            return len(self.subscript)
        if not self.subscript:
            return len(self.superscript)
        return max(len(self.subscript), len(self.superscript))


@dataclass
class AtomSymbol:
    value: str
    type_: str = field(default=ATOM_SYMBOL, init=False)

    @property
    def width(self):
        return len(self.value)


class TextRenderer:
    """
    Used to render atom titles as images, supports superscript for atom charges
    and subscripts for indices.

    Regular text: tokens of 1-2 letters starting with capital letter "H", "Br"
    Subscript notation: in square brackets after underscore "_[1]"
    Superscript notation: in square brackets after caret sign "^[+]"
    """
    def __init__(self, config: TextRendererConfig):
        self.config = config
        font_path = config.font_path
        self.main_font = ImageFont.truetype(str(font_path), config.font_size)
        self.subscript_font = ImageFont.truetype(str(font_path), config.subscript_size)
        self.superscript_font = ImageFont.truetype(str(font_path), config.superscript_size)

    def tokenize(self, text):
        split_by_atom = re.split(re.compile(ATOM_REGEX), text)
        return [self._classify(token) for token in split_by_atom if token]

    def draw_atom(self, offset, drawing, symbol: AtomSymbol):
        drawing.text((offset, 0), symbol.value, font=self.main_font, fill=0)
        return symbol.width * self.config.main_spacing

    def draw_special_symbol(self, offset, drawing, symbol: SpecialSymbol):
        subs_offset = 0
        sups_offset = 0
        if symbol.subscript:
            subs_offset = symbol.width * self.config.special_spacing
            drawing.text((offset, 10), symbol.subscript, font=self.subscript_font, fill=0)
        if symbol.superscript:
            sups_offset = symbol.width * self.config.special_spacing
            drawing.text((offset, 0), symbol.superscript, font=self.superscript_font, fill=0)
        return max(subs_offset, sups_offset)

    def draw_symbol(
        self, offset: int, drawing, symbol: Union[AtomSymbol, SpecialSymbol]
    ) -> int:
        if symbol.type_ == ATOM_SYMBOL:
            return self.draw_atom(offset, drawing, symbol)
        elif symbol.type_ == SPECIAL_SYMBOL:
            return self.draw_special_symbol(offset, drawing, symbol)
        else:
            raise ValueError(f'Unknown symbol type "{symbol.type_}"')

    def get_image(self, text: str, canvas_size=DEFAULT_CANVAS_SIZE):
        tokens = self.tokenize(text)
        text_as_img = Image.new("L", canvas_size, color=255)
        drawing = ImageDraw.Draw(text_as_img)
        offset = 0
        for token in tokens:
            offset += self.draw_symbol(offset, drawing, token)
        text_as_img = np.array(text_as_img)
        # FIXME: `extract_content` is kind of overkill, but for the temporary
        #  solution it will work
        return extract_content(text_as_img)

    @staticmethod
    def _classify(token: str):
        if re.match(re.compile(ATOM_REGEX), token):
            return AtomSymbol(token)
        m = re.search(re.compile(SPECIAL_SYMBOL_REGEX), token)
        if m:
            subs = m.group(3)
            sups = m.group(5)
            return SpecialSymbol(subs, sups)
        raise ValueError(f'Unknown token "{token}"')


if __name__ == '__main__':
    # Example
    conf = TextRendererConfig('fonts/Inconsolata-Regular.ttf')
    text_renderer = TextRenderer(config=conf)
    img = text_renderer.get_image('NH_[2]^[-2]NH_[2]^[+2]')
    save_img('rendered_text.png', img)
