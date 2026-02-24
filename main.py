#!/usr/bin/env python3
"""
AutoBot - ReAct Agent
Main entry point for the ReAct (Reason + Act) agent system.
"""

import asyncio
import logging
import yaml
from core.react_orchestrator import ReActOrchestrator


def load_config():
    """Load configuration from YAML."""
    with open('config/settings.yaml', 'r') as f:
        return yaml.safe_load(f)


def setup_logging(config):
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, config['logging']['level']),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config['logging']['file']),
            logging.StreamHandler()
        ]
    )


async def main():
    """Main entry point."""
    config = load_config()
    setup_logging(config)

    logger = logging.getLogger(__name__)
    logger.info("Starting AutoBot ReAct Agent...")

    # Initialize ReAct orchestrator
    orchestrator = ReActOrchestrator(config)
    
    if not await orchestrator.initialize():
        logger.error("Failed to initialize ReAct orchestrator")
        return

    # Interactive loop
    print("\n" + "="*60)
    print("AutoBot ReAct Agent v0.3.0")
    print("="*60)
    print("Type 'quit' to exit, 'history' for previous conversations, 'clear' to clear session")
    print("="*60 + "\n")

    try:
        while True:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                # Flush short-term to long-term before exit
                try:
                    result = await orchestrator.memory.flush_short_to_long_term()
                    print(f"Flushed short-term to long-term: {result}\n")
                except Exception:
                    print("Error flushing short-term memory before exit.")
                print("Goodbye!")
                break

            if user_input.lower() == 'history':
                # Show recent long-term and short-term interactions
                try:
                    long_hist = await orchestrator.memory.get_recent_interactions(limit=50)
                    short_hist = await orchestrator.memory.get_short_term_interactions(limit=50)
                    print("\n--- Long-term (most recent) ---")
                    for item in long_hist:
                        ts = item.get('timestamp')
                        print(f"User: {item.get('user_input')}\nAssistant: {item.get('response')}\n")
                    print("\n--- Short-term (current session) ---")
                    for item in short_hist:
                        print(f"User: {item.get('user_input')}\nAssistant: {item.get('response')}\n")
                    print("")
                except Exception as exc:
                    print(f"Error retrieving history: {exc}")
                continue

            if user_input.lower() == 'clear':
                # Append short-term to long-term and clear
                try:
                    result = await orchestrator.memory.flush_short_to_long_term()
                    orchestrator.chat_history.clear()
                    print(f"Short-term cleared and appended to long-term: {result}\n")
                except Exception as exc:
                    print(f"Error clearing short-term: {exc}")
                continue
            
            # Process with ReAct
            print("\n[ReAct processing...]")
            response = await orchestrator.handle_input(user_input)
            print(f"\nAutoBot: {response}\n")
            # Print post-message help block
            print("============================================================")
            print("Type 'quit' to exit, 'history' for previous conversations, 'clear' to clear history")
            print("============================================================\n")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")
    except Exception as exc:
        logger.exception("Error in main loop: %s", exc)
        print(f"Error: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
