"""CLI runner for PydanticAI Learning Hub examples.

This module provides a unified command-line interface to run all the learning examples.

Usage:
    uv run pydantic-learning --list
    uv run pydantic-learning 01
    uv run pydantic-learning 03 --answer potato
"""

import argparse
import asyncio
import importlib
import sys
from pathlib import Path


def list_examples():
    """List all available examples."""
    examples = [
        ('01', '01_basic_agent', 'Basic Agent - Simplest possible agent'),
        ('02', '02_agent_with_tools', 'Agent with Tools - Adding tool capabilities'),
        ('03', '03_multi_agent', 'Multi-Agent System - Twenty questions game'),
        ('04', '04_durable_dbos', 'Durable Execution with DBOS'),
        ('05', '05_durable_temporal', 'Durable Execution with Temporal'),
        ('06', '06_deep_research', 'Advanced Multi-Agent - Deep research'),
        ('07', '07_deep_research_dbos', 'Durable Deep Research with DBOS'),
        ('08', '08_deep_research_temporal', 'Durable Deep Research with Temporal'),
        ('09', '09_evals', 'Evaluation Patterns'),
    ]

    print('\nAvailable Examples:\n')
    for num, module, description in examples:
        print(f'  {num}: {description}')
        print(f'      Run with: uv run pydantic-learning {num}')
        print()


def run_example(example_num: str, **kwargs):
    """Run a specific example by number.

    Args:
        example_num: The example number (e.g., '01', '02', '03')
        **kwargs: Additional arguments to pass to the example
    """
    # Map example numbers to module names
    examples_map = {
        '01': '01_basic_agent',
        '02': '02_agent_with_tools',
        '03': '03_multi_agent',
        '04': '04_durable_dbos',
        '05': '05_durable_temporal',
        '06': '06_deep_research',
        '07': '07_deep_research_dbos',
        '08': '08_deep_research_temporal',
        '09': '09_evals',
    }

    if example_num not in examples_map:
        print(f'Error: Example {example_num} not found.')
        print('Use --list to see available examples.')
        sys.exit(1)

    module_name = examples_map[example_num]
    print(f'Running example {example_num}: {module_name}...\n')

    try:
        # Import the module
        module = importlib.import_module(f'pydantic_learning.agents.{module_name}')

        # Run the main function if it exists
        if hasattr(module, 'main'):
            if asyncio.iscoroutinefunction(module.main):
                asyncio.run(module.main(**kwargs))
            else:
                module.main(**kwargs)
        elif hasattr(module, 'play'):
            # For multi-agent examples that use play() function
            answer = kwargs.get('answer', 'potato')
            asyncio.run(module.play(answer))
        else:
            print(f'Error: Module {module_name} has no main() or play() function.')
            sys.exit(1)

    except ImportError as e:
        print(f'Error importing module: {e}')
        print('Make sure all dependencies are installed.')
        sys.exit(1)
    except Exception as e:
        print(f'Error running example: {e}')
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='PydanticAI Learning Hub - Run examples and tutorials',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pydantic-learning --list              List all available examples
  pydantic-learning 01                  Run basic agent example
  pydantic-learning 03 --answer potato  Run multi-agent with custom answer
        """,
    )

    parser.add_argument(
        'example',
        nargs='?',
        help='Example number to run (e.g., 01, 02, 03)',
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available examples',
    )

    parser.add_argument(
        '--answer',
        type=str,
        help='Answer for multi-agent examples (default: potato)',
    )

    args = parser.parse_args()

    if args.list:
        list_examples()
    elif args.example:
        kwargs = {}
        if args.answer:
            kwargs['answer'] = args.answer
        run_example(args.example, **kwargs)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
