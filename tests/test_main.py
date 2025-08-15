"""Tests for the main module."""

import pytest

from python_uv_template.main import greet, main, process_names


class TestGreet:
    """Tests for the greet function."""

    def test_greet_default(self) -> None:
        """Test greeting with default parameters."""
        result = greet("World")
        assert result == "Hello, World!"

    def test_greet_custom_greeting(self) -> None:
        """Test greeting with custom greeting."""
        result = greet("Python", "Welcome")
        assert result == "Welcome, Python!"

    def test_greet_empty_name(self) -> None:
        """Test greeting with empty name."""
        result = greet("")
        assert result == "Hello, !"

    @pytest.mark.parametrize(
        "name,greeting,expected",
        [
            ("Alice", "Hi", "Hi, Alice!"),
            ("Bob", "Hey", "Hey, Bob!"),
            ("Charlie", "Hello", "Hello, Charlie!"),
        ],
    )
    def test_greet_parametrized(self, name: str, greeting: str, expected: str) -> None:
        """Test greeting with multiple combinations."""
        result = greet(name, greeting)
        assert result == expected


class TestProcessNames:
    """Tests for the process_names function."""

    def test_process_names_empty_list(self) -> None:
        """Test processing empty list."""
        result = process_names([])
        assert result == []

    def test_process_names_single_name(self) -> None:
        """Test processing single name."""
        result = process_names(["Alice"])
        assert result == ["Hello, Alice!"]

    def test_process_names_multiple_names(self) -> None:
        """Test processing multiple names."""
        names = ["Alice", "Bob", "Charlie"]
        result = process_names(names)
        expected = ["Hello, Alice!", "Hello, Bob!", "Hello, Charlie!"]
        assert result == expected

    def test_process_names_custom_greeting(self) -> None:
        """Test processing with custom greeting."""
        names = ["Alice", "Bob"]
        result = process_names(names, "Hi")
        expected = ["Hi, Alice!", "Hi, Bob!"]
        assert result == expected


class TestMain:
    """Tests for the main function."""

    def test_main_returns_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that main returns 0."""
        result = main()
        assert result == 0

        # Verify that something is printed
        captured = capsys.readouterr()
        assert "Python UV Template started!" in captured.out
        assert "Application executed successfully!" in captured.out
