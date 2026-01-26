"""Entry point for running copilot_conductor as a module.

Usage:
    python -m copilot_conductor
"""

from copilot_conductor.cli.app import app


def main() -> None:
    """Main entry point for the copilot-conductor CLI."""
    app()


if __name__ == "__main__":
    main()
