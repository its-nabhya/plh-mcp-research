from src.worlds.micro_world.world import MicroWorld
from src.agent.agent import Agent
from src.logging.validator import Validator
from src.logging.logger import Logger


def run():
    world = MicroWorld()
    agent = Agent()
    validator = Validator(world.tools)
    logger = Logger()

    prompts = [
        "Add 5 and 3",
        "Subtract 5 and 3",  # will hallucinate
        "Add two numbers",
        "Use advanced calculator"
    ]

    for prompt in prompts:
        print(f"\nPrompt: {prompt}")

        # Agent step
        agent_output = agent.act(prompt)
        reasoning = agent_output["reasoning"]
        tool_call = agent_output["tool_call"]

        # World execution
        response = world.execute(tool_call)

        # Validation
        validation = validator.validate(tool_call)

        # Logging
        log_entry = logger.log(
            world="micro_world",
            model="dummy-agent",
            prompt=prompt,
            reasoning=reasoning,
            tool_call=tool_call,
            validation=validation,
            response=response
        )

        print("Tool Call:", tool_call)
        print("Validation:", validation)
        print("Response:", response)


if __name__ == "__main__":
    run()