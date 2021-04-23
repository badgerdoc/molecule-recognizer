from recognizer.drawing.text import TextRenderer, TextRendererConfig


def test_tokenize():
    conf = TextRendererConfig('./resources/fonts/Inconsolata-Regular.ttf')
    text_renderer = TextRenderer(config=conf)
    tokens = text_renderer.tokenize('NeH_[2]^[-2]')
    assert tokens[0].value == 'Ne'
    assert tokens[1].value == 'H'
    assert tokens[2].superscript == '-2'
    assert tokens[2].subscript == '2'
    text_renderer.get_image('NH_[2]^[-2]NH_[2]^[+2]')
    # TODO: add assertions
