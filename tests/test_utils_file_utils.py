import os
import tempfile

from model_convert.utils.file_utils import read_lists, read_symbol_table


class TestReadLists:
    def test_basic(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("line1\nline2\nline3\n")
            path = f.name
        try:
            result = read_lists(path)
            assert result == ["line1", "line2", "line3"]
        finally:
            os.unlink(path)

    def test_strips_whitespace(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("  hello  \n  world  \n")
            path = f.name
        try:
            result = read_lists(path)
            assert result == ["hello", "world"]
        finally:
            os.unlink(path)

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("")
            path = f.name
        try:
            result = read_lists(path)
            assert result == []
        finally:
            os.unlink(path)


class TestReadSymbolTable:
    def test_basic(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("<blank> 0\nhello 1\nworld 2\n")
            path = f.name
        try:
            result = read_symbol_table(path)
            assert result == {"<blank>": 0, "hello": 1, "world": 2}
        finally:
            os.unlink(path)

    def test_single_entry(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("token 42\n")
            path = f.name
        try:
            result = read_symbol_table(path)
            assert result == {"token": 42}
        finally:
            os.unlink(path)
