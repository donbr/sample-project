"""
CLI runner for PydanticAI learning examples.

Usage:
    uv run python -m pydantic_learning --help
    uv run python -m pydantic_learning list
    uv run python -m pydantic_learning run 05
    uv run python -m pydantic_learning run 05 --input "banana"
"""
import argparse
import asyncio
import sys
from pathlib import Path

# Import the actual agent modules from src/agents
try:
    # Add src to path to import agents
    src_path = Path(__file__).parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from agents import twenty_questions
    from agents import deep_research
    from agents import twenty_questions_dbos
    from agents import twenty_questions_temporal
    from agents import deep_research_dbos
    from agents import deep_research_temporal
    from agents import twenty_questions_evals
except ImportError as e:
    print(f"Warning: Could not import agents: {e}")


# Map lesson numbers to available agents
LESSONS = {
    "05": ("twenty_questions", "Multi-Agent: Twenty Questions", twenty_questions),
    "06": ("deep_research", "Multi-Agent: Deep Research", deep_research),
    "07": ("twenty_questions_dbos", "Durable Execution: Twenty Questions (DBOS)", twenty_questions_dbos),
    "08": ("twenty_questions_temporal", "Durable Execution: Twenty Questions (Temporal)", twenty_questions_temporal),
    "09": ("twenty_questions_evals", "Agent Evaluations: Twenty Questions", twenty_questions_evals),
    "10": ("deep_research_dbos", "Durable Execution: Deep Research (DBOS)", deep_research_dbos),
    "11": ("deep_research_temporal", "Durable Execution: Deep Research (Temporal)", deep_research_temporal),
}


def list_lessons():
    """Print available lessons."""
    print("\n📚 Available Lessons:\n")
    print(f"{'#':<4} {'Module':<30} {'Description'}")
    print("-" * 80)
    for num, (module_name, desc, _) in LESSONS.items():
        print(f"{num:<4} {module_name:<30} {desc}")
    print()
    print("Note: Lessons 01-04 are planned but not yet implemented.")
    print()


def run_lesson(lesson_num: str, user_input: str = None):
    """Run a specific lesson."""
    if lesson_num not in LESSONS:
        print(f"❌ Unknown lesson: {lesson_num}")
        print(f"   Available: {', '.join(LESSONS.keys())}")
        sys.exit(1)

    module_name, desc, module = LESSONS[lesson_num]
    print(f"\n🎓 Running Lesson {lesson_num}: {desc}\n")
    print("-" * 60)

    try:
        # Different agents have different entry points
        if hasattr(module, 'main'):
            # If module has a main() function
            if asyncio.iscoroutinefunction(module.main):
                if user_input:
                    asyncio.run(module.main(user_input))
                else:
                    asyncio.run(module.main())
            else:
                if user_input:
                    module.main(user_input)
                else:
                    module.main()
        elif hasattr(module, 'play'):
            # For twenty_questions style modules
            answer = user_input or "potato"
            print(f"Playing with answer: {answer}")
            asyncio.run(module.play(answer))
        elif hasattr(module, 'run_research'):
            # For deep_research style modules
            topic = user_input or "What are the latest developments in PydanticAI?"
            print(f"Researching: {topic}")
            asyncio.run(module.run_research(topic))
        else:
            print(f"⚠️  Module {module_name} doesn't have a recognized entry point")
            print(f"   Available attributes: {[attr for attr in dir(module) if not attr.startswith('_')]}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PydanticAI Learning Hub CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all lessons
  uv run python -m pydantic_learning list

  # Run Twenty Questions with default answer
  uv run python -m pydantic_learning run 05

  # Run Twenty Questions with custom answer
  uv run python -m pydantic_learning run 05 --input "banana"

  # Run Deep Research with custom topic
  uv run python -m pydantic_learning run 06 --input "PydanticAI best practices"
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List command
    subparsers.add_parser("list", help="List available lessons")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a lesson")
    run_parser.add_argument("lesson", help="Lesson number (05-11)")
    run_parser.add_argument(
        "--input", "-i",
        help="Input for the lesson (answer for Twenty Questions, topic for Deep Research)"
    )

    args = parser.parse_args()

    if args.command == "list":
        list_lessons()
    elif args.command == "run":
        run_lesson(args.lesson, args.input)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
