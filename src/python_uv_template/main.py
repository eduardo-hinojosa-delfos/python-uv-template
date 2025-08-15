"""Main application module."""

import sys


def greet(name: str, greeting: str = "Hello") -> str:
    """
    Generate a personalized greeting.

    Args:
        name: The name of the person to greet
        greeting: The greeting to use (default "Hello")

    Returns:
        A formatted greeting message

    Example:
        >>> greet("World")
        'Hello, World!'
        >>> greet("Python", "Welcome")
        'Welcome, Python!'
    """
    return f"{greeting}, {name}!"


def process_names(names: list[str], greeting: str | None = None) -> list[str]:
    """
    Process a list of names and generate greetings.

    Args:
        names: List of names to process
        greeting: Optional greeting to use

    Returns:
        List of greeting messages
    """
    if not names:
        return []

    default_greeting = greeting or "Hello"
    return [greet(name, default_greeting) for name in names]


def main() -> int:
    """
    Main application function.

    Returns:
        Exit code (0 for success)
    """
    print("🚀 Python UV Template started!")

    # Example usage
    names = ["World", "Python", "Developer"]
    greetings = process_names(names)

    for greeting in greetings:
        print(greeting)

    print("✅ Application executed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
