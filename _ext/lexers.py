"""Подсветка кода ноутбуков, у которых в метаданных записан лексер ipython2 (так сохраняет DataSpell)."""
from IPython.lib.lexers import IPython3Lexer


def setup(app):
    app.add_lexer("ipython2", IPython3Lexer)
    return {"parallel_read_safe": True}
