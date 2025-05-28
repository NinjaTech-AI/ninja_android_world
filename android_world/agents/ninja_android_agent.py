"""Ninja Android agent, fast scaffold based on UI-TARS"""

from dotenv import load_dotenv

load_dotenv(override=True)

import asyncio
import base64
import json
import logging
import io
import os
import re
import sys
import time
import numpy as np

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from android_world.agents import base_agent
from android_world.env import interface
from android_world.env import adb_utils

from openai import AsyncOpenAI, OpenAI
from PIL import Image

from android_world.agents.ninja_android_agent_controller import (
    ADBConnection,
    back,
    backspace,
    delete,
    enter,
    get_screenshot,
    get_xml,
    home,
    open_app_with_name,
    swipe,
    tap,
    type,
    volume_down,
    volume_up,
)

from android_world.agents.ninja_android_agent_xml_extractor import rule_based_extractor



## setup log
logger = logging.getLogger("agent_ui_tars")
logger.setLevel(logging.DEBUG)
# Global formatter definitions
stdout_formatter = logging.Formatter(
    fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s"
)
file_formatter = logging.Formatter(
    fmt="[%(asctime)s %(levelname)s %(module)s/%(lineno)d-%(processName)s] %(message)s"
)


# have new log files given log directory (e.g. set based task id )
def setup_logging(log_dir):
    # Store the stdout handler if it exists
    stdout_handler = None
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
            stdout_handler = handler
        logger.removeHandler(handler)

    # Recreate stdout handler if it didn't exist before
    if stdout_handler is None:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.INFO)
        stdout_handler.setFormatter(stdout_formatter)

    # Add back the stdout handler
    logger.addHandler(stdout_handler)

    log_dir.mkdir(parents=True, exist_ok=True)

    # Create new handlers
    file_info_handler = logging.FileHandler(log_dir / "info.log", encoding="utf-8")

    file_debug_handler = logging.FileHandler(log_dir / "debug.log", encoding="utf-8")

    file_info_handler.setLevel(logging.INFO)
    file_debug_handler.setLevel(logging.DEBUG)

    file_info_handler.setFormatter(file_formatter)
    file_debug_handler.setFormatter(file_formatter)

    # Add new handlers
    logger.addHandler(file_info_handler)
    logger.addHandler(file_debug_handler)


ACTION_SPCAE = """\
click(start_box='<|box_start|>(x1, y1)<|box_end|>')
long_press(start_box='<|box_start|>(x1,y1)<|box_end|>', time='')
type(content='') # If you want to submit your input, use "\\n" at the end of `content`.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
open_app(app_name=\'\')
press_home()
press_back()
hotkey(key='') # The available keys: enter,back,home,backspace
wait()
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.\
"""

MOBILE_PROMPT_TEMPLATE = """\
You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
```
Thought: ...
Action: ...
```

## Action Space
{action_space}

## Note
- Use {language} in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}"""

MEMORY_TEMPLATE_INITIAL = """\
You are a helpful agent that summarizes the image content. You are given a task with an Android phone screenshot. Your task is to summarize the key information from the screenshot that is related to the task.

## Output Format
Screenshot Description: ...

## Task
{instruction}"""

MEMORY_TEMPLATE = """\
You are a helpful agent that summarizes the image content. You are given a task with a thought, an action, and an Android phone screenshot. Your task is to summarize the key information from the screenshot that is related to the task, thought, and action.

## Output Format
Screenshot Description: ...

## Task
{instruction}

## Thought
{thought}

## Action
{action}"""

# define parameters
IMAGE_BUFFER_SIZE = 10
# Hardcoded screen resolution
WIDTH = 1080
HEIGHT = 2400
# Reduce image size to reduce token usage of UI-TARS and the general model
RESIZE_FACTOR = 0.5

def base64_encode_image(image_source, resize_factor: float = RESIZE_FACTOR):
    """Encode image to base64 string with optional resizing.
    
    Args:
        image_source: Either a path to image file (str/Path) or numpy array (state.pixels)
        resize_factor: Factor to resize image by (default RESIZE_FACTOR). If 1, no resize is performed.
        
    Returns:
        str: Base64 encoded image string
        
    Raises:
        ValueError: If image_source is invalid or image processing fails
    """
    try:
        # Handle numpy array (state.pixels)
        if isinstance(image_source, np.ndarray):
            img = Image.fromarray(image_source)
        # Handle file path
        elif isinstance(image_source, (str, Path)):
            img = Image.open(image_source)
        else:
            raise ValueError(f"Unsupported image source type: {type(image_source)}")

        # Convert to RGB if needed
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        
        # Only resize if factor is not 1
        if resize_factor != 1:
            # Calculate new dimensions
            width, height = img.size
            new_width = int(width * resize_factor)
            new_height = int(height * resize_factor)
            
            # Resize image
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Save to bytes buffer
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=95)
        
        # Encode to base64
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    except Exception as e:
        raise ValueError(f"Failed to process image: {str(e)}")


def add_box_token(input_string) -> str:
    # Step 1: Split the string into individual actions
    if "Action: " in input_string and "start_box=" in input_string:
        suffix = input_string.split("Action: ")[0] + "Action: "
        actions = input_string.split("Action: ")[1:]
        processed_actions = []
        for action in actions:
            action = action.strip()
            # Step 2: Extract coordinates (start_box or end_box) using regex
            coordinates = re.findall(
                r"(start_box|end_box)='\((\d+),\s*(\d+)\)'", action
            )

            updated_action = action  # Start with the original action
            for coord_type, x, y in coordinates:
                # Convert x and y to integers
                updated_action = updated_action.replace(
                    f"{coord_type}='({x},{y})'",
                    f"{coord_type}='<|box_start|>({x},{y})<|box_end|>'",
                )
            processed_actions.append(updated_action)

        # Step 5: Reconstruct the final string
        final_string = suffix + "\n\n".join(processed_actions)
    else:
        final_string = input_string
    return final_string


def escape_single_quotes(text: str):
    pattern = r"(?<!\\)'"
    return re.sub(pattern, r"\\'", text)

# execute action directory through adb command.
def execute_action_adb(
    adb_connection: ADBConnection,
    action: str
) -> tuple[bool, str | None, str | None]:
    """
    Parses the response string and executes the corresponding function.
    Returns:
        tuple[bool, str | None, str | None]:
            - bool: True if action was successfully executed, False otherwise
            - str | None: Action name if parsed successfully, None otherwise
            - str | None: Error message if parsing failed, None if parsing successful (regardless of execution success)
    """

    # Parse the function call
    try:
        if "click" in action:
            coord_match = re.search(r"start_box='\((\d+),(\d+)\)'", action)
            if coord_match:
                # Extract and normalize coordinates
                x = int(coord_match.group(1))
                y = int(coord_match.group(2))

                # Normalize coordinates
                normalized_x = int(x * WIDTH / 1000)
                normalized_y = int(y * HEIGHT / 1000)

                action_name = f"tap({normalized_x}, {normalized_y})"
                tap(adb_connection, normalized_x, normalized_y)
                return True, action_name, None
            else:
                return False, None, "Could not parse coordinates from action"

        elif "long_press" in action:
            # Extract coordinates from the parameters
            coord_match = re.search(r"start_box='\((\d+),(\d+)\)'", action)
            if coord_match:
                x = int(coord_match.group(1))
                y = int(coord_match.group(2))

                # Normalize coordinates
                normalized_x = int(x * WIDTH / 1000)
                normalized_y = int(y * HEIGHT / 1000)

                # For long press, start and end coordinates are the same
                action_name = f"swipe({normalized_x}, {normalized_y}, {normalized_x}, {normalized_y})"
                swipe(
                    adb_connection,
                    normalized_x,
                    normalized_y,
                    normalized_x,
                    normalized_y,
                    time=1500,
                )
                return True, action_name, None
            else:
                return False, None, "Could not parse coordinates from long_press action"

        elif "scroll" in action:
            start_coord_match = re.search(r"start_box='\((\d+),(\d+)\)'", action)
            end_coord_match = re.search(r"end_box='\((\d+),(\d+)\)'", action)
            if start_coord_match and end_coord_match:
                x_start = int(start_coord_match.group(1))
                y_start = int(start_coord_match.group(2))
                x_end = int(end_coord_match.group(1))
                y_end = int(end_coord_match.group(2))

                # Normalize coordinates
                x1 = int(x_start * WIDTH / 1000)
                y1 = int(y_start * HEIGHT / 1000)

                x2 = int(x_end * WIDTH / 1000)
                y2 = int(y_end * HEIGHT / 1000)

                action_name = f"swipe({x1}, {y1}, {x2}, {y2})"
                swipe(adb_connection, x1, y1, x2, y2)
                return True, action_name, None

            return False, None, "Could not parse coordinates from drag action"

        elif "finished" in action:
            content_match = re.search(r"content=['\"]([^'\"]+)['\"]", action)
            if content_match:
                content = content_match.group(1)
                return True, f"finish({content})", None
            else:
                return True, "finish", None

        elif "press_home()" in action:
            action_name = "home()"
            home(adb_connection)
            return True, action_name, None

        elif "press_back()" in action:
            action_name = "back()"
            back(adb_connection)
            return True, action_name, None

        elif "open_app" in action:
            # Extract app name from the parameters
            app_name_match = re.search(r"app_name=['\"]([^'\"]+)['\"]", action)
            if app_name_match:
                app_name = app_name_match.group(1)

                # hardcode for now
                if "music" in app_name.lower():
                    app_name = "music"
                elif "book" in app_name.lower():
                    app_name = "booking"
                elif "news" in app_name.lower():
                    app_name = "news"
                elif "note" in app_name.lower():
                    app_name = "notes"

                action_name = f"open_app_with_name({app_name})"
                try:
                    success = open_app_with_name(adb_connection, app_name)
                    return True, action_name, None
                except RuntimeError as e:
                    return False, action_name, str(e)
            else:
                return False, None, "Could not parse app name from action"

        elif "type(content" in action:

            def escape_quotes(match):
                content = match.group(1)  # get content value
                return content

            pattern = r"type\(content='(.*?)'\)"  # match type(content='...')
            content = re.sub(pattern, escape_quotes, action)

            # process string
            text = escape_single_quotes(content)
            print("raw text for typing is:", repr(text))

            action_name = f"type({text})"
            type(adb_connection, text)

            if text.endswith("\n"):
                enter(adb_connection)

            return True, action_name, None

        elif "wait" in action:
            time.sleep(2)
            return True, "wait()", None

        elif "hotkey" in action:
            if "enter" in action:
                enter(adb_connection)
                return True, "hotkey(key='enter')", None
            elif "back" in action:
                back(adb_connection)
                return True, "hotkey(key='back')", None
            elif "home" in action:
                home(adb_connection)
                return True, "hotkey(key='home')", None
            elif "backspace" in action:
                backspace(adb_connection)
                return True, "hotkey(key='backspace')", None
            elif "delete" in action:
                delete(adb_connection)
                return True, "hotkey(key='delete')", None
            elif "volume_up" in action:
                volume_up(adb_connection)
                return True, "hotkey(key='volume_up')", None
            elif "volume_down" in action:
                volume_down(adb_connection)
                return True, "hotkey(key='volume_down')", None
            else:
                return False, None, f"Action {action} not recognized."
        else:
            return False, None, f"Action {action} not recognized."

    except Exception as e:
        return False, None, f"Error parsing action: {e}"

@dataclass
class ChatResponse:
    mode: str
    response: str


async def generate_completion(
    client, model_name: str, messages, temperature
) -> ChatResponse:
    if "ui-tars" in model_name.lower():
        mode = "worker"
    else:
        mode = "memory"

    # create streaming response
    completion = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
    )
    prediction = completion.choices[0].message.content.strip()

    return ChatResponse(mode=mode, response=prediction)


def extract_thought(prediction: str) -> str:
    if prediction.startswith("Thought:"):
        thought_pattern = r"Thought: (.+?)(?=\s*Action:|$)"

        thought_match = re.search(thought_pattern, prediction, re.DOTALL)
        if thought_match:
            if len(thought_match.groups()) == 1:
                thought = thought_match.group(1).strip()
                return thought
        else:
            raise ValueError("thought not found")


def extract_action(prediction: str) -> str:
    if "Action: " in prediction:
        action = prediction.split("Action: ", 1)[1].strip()
        return action
    else:
        raise ValueError("action not found")


def get_top_actions(predictions: list[str]):
    action_prediction_map = {}
    for prediction in predictions:
        action = extract_action(prediction)
        action_prediction_map[action] = prediction

    counter = Counter(action_prediction_map.keys())
    if not counter:
        return [], action_prediction_map

    max_count = max(counter.values())
    return [
        action for action, count in counter.items() if count == max_count
    ], action_prediction_map


def revert_coordinate(screen_x: int, screen_y: int):
    model_x = int(screen_x * 1000 / WIDTH)
    model_y = int(screen_y * 1000 / HEIGHT)
    return model_x, model_y

class NinjaAndroidAgent(base_agent.EnvironmentInteractingAgent):
    """Ninja Android agent, fast scaffold based on UI-TARS"""

    def __init__(
            self,
            env: interface.AsyncEnv,
            base_url: str = "https://api.runpod.ai/v2/vn71d606be7ixc/openai/v1",
            ui_tar_model_name: str = "bytedance-research/UI-TARS-72B-DPO",
            name: str = 'NinjaAgent',
            transition_pause: float = 5.0
        ):
            """Initialize NinjaAndroidAgent.

            Args:
                env: The environment
                base_url: API base URL for UI-TARS model
                ui_tar_model_name: Model name for UI understanding
                name: Agent name
                transition_pause: Seconds to wait after action
            """
            super().__init__(env, name, transition_pause)

            # Initialize agents
            self.worker_async_agent = AsyncOpenAI(
                base_url=base_url,
                api_key=os.environ.get("UI_TARS_API_KEY"),
            )
            self.general_async_agent = AsyncOpenAI()
            self.general_sync_agent = OpenAI()
            
            # Store parameters
            self.ui_tar_model_name = ui_tar_model_name
                        
            self.adb_connection = ADBConnection()
            # check the connection, not needed for Android world??
            #if self.adb_connection.check_adb_connection_with_reconnect() is False:
            #    print("problem connecting with the android phone")
            #    return
            #else:
            #    print("successfully connected with the android phone")

    def reset(self, go_home: bool = True) -> None:
            """Resets the agent state."""
            super().reset(go_home)
            
            # Reset history
            self.history_base64_images = []
            self.history_image_memory: list[str] = []
            self.history_actions: list[str] = []
            self.history_thoughts: list[str] = []
            self.history_predictions: list[str] = []
            self.messages: list[dict] = []
            self.steps_info = []
            self.recorded_time = []

            # Todo, passing more task information so that easier visual log can be added here 
            task_id = time.strftime("%Y%m%d-%H%M%S")
            self.log_dir = Path("logs")/ task_id
            
            # create a directory to store the screenshots
            self.screenshot_log_path = self.log_dir / "screenshots"
            print(f'screenshot_log_path={self.screenshot_log_path}')
            self.screenshot_log_path.mkdir(parents=True, exist_ok=True)

    def execute_action_env(self, action: str) -> tuple[bool, str | None, str | None]:
        """Executes action using environment interface.
        
        Args:
            action: Action string in NinjaAndroid format
            
        Returns:
            tuple[bool, str | None, str | None]:
                - bool: True if action was successfully executed
                - str | None: Action name if parsed successfully
                - str | None: Error message if failed
        """
        try:
            # Get screen dimensions
            WIDTH, HEIGHT = self.env.logical_screen_size

            if "click" in action:
                coord_match = re.search(r"start_box='\((\d+),(\d+)\)'", action)
                if coord_match:
                    x = int(coord_match.group(1))
                    y = int(coord_match.group(2))
                    
                    # Normalize coordinates from 1000-based to actual screen dimensions
                    normalized_x = int(x * WIDTH / 1000)
                    normalized_y = int(y * HEIGHT / 1000)
                    
                    adb_utils.tap_screen(normalized_x, normalized_y, self.env.controller)
                    return True, f"tap({normalized_x}, {normalized_y})", None

            elif "type" in action:
                match = re.search(r"type\(content='(.*?)'\)", action)
                if match:
                    text = match.group(1)
                    adb_utils.type_text(text, self.env.controller)
                    if text.endswith("\n"):
                        adb_utils.press_enter_button(self.env.controller)
                    return True, f"type({text})", None

            elif "scroll" in action:
                start_coord_match = re.search(r"start_box='\((\d+),(\d+)\)'", action)
                end_coord_match = re.search(r"end_box='\((\d+),(\d+)\)'", action)
                if start_coord_match and end_coord_match:
                    x1 = int(start_coord_match.group(1))
                    y1 = int(start_coord_match.group(2))
                    x2 = int(end_coord_match.group(1))
                    y2 = int(end_coord_match.group(2))
                    
                    # Normalize coordinates
                    normalized_x1 = int(x1 * WIDTH / 1000)
                    normalized_y1 = int(y1 * HEIGHT / 1000)
                    normalized_x2 = int(x2 * WIDTH / 1000)
                    normalized_y2 = int(y2 * HEIGHT / 1000)
                    
                    command = adb_utils.generate_swipe_command(
                        normalized_x1, normalized_y1, 
                        normalized_x2, normalized_y2
                    )
                    adb_utils.issue_generic_request(command, self.env.controller)
                    return True, f"swipe({normalized_x1}, {normalized_y1}, {normalized_x2}, {normalized_y2})", None

            elif "long_press" in action:
                coord_match = re.search(r"start_box='\((\d+),(\d+)\)'", action)
                if coord_match:
                    x = int(coord_match.group(1))
                    y = int(coord_match.group(2))
                    
                    # Normalize coordinates
                    normalized_x = int(x * WIDTH / 1000)
                    normalized_y = int(y * HEIGHT / 1000)
                    
                    adb_utils.long_press(normalized_x, normalized_y, self.env.controller)
                    return True, f"long_press({normalized_x}, {normalized_y})", None

            elif "finished" in action:
                return True, "finish", None

            elif "press_home" in action:
                adb_utils.press_home_button(self.env.controller)
                return True, "home", None

            elif "press_back" in action:
                adb_utils.press_back_button(self.env.controller)
                return True, "back", None

            elif "open_app" in action:
                match = re.search(r"app_name=['\"]([^'\"]+)['\"]", action)
                if match:
                    app_name = match.group(1)
                    adb_utils.launch_app(app_name, self.env.controller)
                    return True, f"open_app({app_name})", None

            return False, None, f"Action {action} not recognized or not implemented"

        except Exception as e:
            return False, None, f"Error executing action: {str(e)}"

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        """Performs one step of interaction.

        Args:
            goal: The instruction/goal for this step

        Returns:
            AgentInteractionResult containing success status and step data
        """

        instruction = goal
        step_info = {
            "raw_screenshot": None,
            "task": instruction,
            "thought": "",
            "initial_action": "",
            "correct_action": "",
            "memory": "",
            "step_time": 0.0,
            "average_step_time": 0.0,
            "total_time": 0.0,
        }

        # Todo, use the base_agent's supporting function to retrieve the phone state
        # where is also handle the transition consistently??
        state = self.env.get_state(wait_to_stabilize=False)
        # Get screenshot and encode
        screenshot = state.pixels.copy()
        base64_image = base64_encode_image(screenshot)
        step = len(self.steps_info)

         # save the latest phone screenshot (for easier debug)
        Image.fromarray(screenshot).save(
            self.screenshot_log_path / f"{step}.jpg",
            format='JPEG',
            quality=95
        )

        # save the base64 encoded image into history
        self.history_base64_images.append(base64_image)

        if step == 0:
            memory_template = MEMORY_TEMPLATE_INITIAL.format(instruction=instruction)
            # append the initial user query
            self.messages.append(
                {
                    "role": "user",
                    "content": MOBILE_PROMPT_TEMPLATE.format(
                        language="English",
                        instruction=instruction,
                        action_space=ACTION_SPCAE,
                    ),
                }
            )
        else:
            memory_template = MEMORY_TEMPLATE.format(
                instruction=instruction,
                thought=self.history_thoughts[-1],
                action=self.history_actions[-1],
            )
        # append the latest base64 image code into messages
        self.messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    }
                ],
            }
        )

        if len(self.history_base64_images) > IMAGE_BUFFER_SIZE:
            # remove the second item in the messages that contains the role "user"
            for i in range(1, len(self.messages)):
                if self.messages[i]["role"] == "user" and self.messages[i]["content"][0]["type"] == "image_url":
                    del self.messages[i]
                    break
 # record the step time
        step_time_duration = 0.0

        # model prediction
        async def generate_all_completions():
            tasks = []
            chat_history = []
            for model_name in [self.ui_tar_model_name, "gpt-4o"]:
                if model_name == "gpt-4o":
                    client = self.general_async_agent
                    temperature = 0.0
                    num_task = 1
                    chat_history = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": memory_template,
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    },
                                },
                            ],
                        }
                    ]
                else:
                    client = self.worker_async_agent
                    temperature = 0.2
                    num_task = 2
                    chat_history = self.messages

                # add async tasks into the list
                for _ in range(num_task):
                    task = generate_completion(
                        client=client,
                        model_name=model_name,
                        messages=chat_history,
                        temperature=temperature,
                    )
                    tasks.append(task)

            # Run all tasks concurrently and gather results
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results

        # run the async tasks in the event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        worker_start_time = time.time()
        # process results
        chat_responses: list[ChatResponse] = loop.run_until_complete(
            generate_all_completions()
        )

        predictions: list[str] = []
        for response in chat_responses:
            # Add exception handling
            if isinstance(response, Exception):
                print(f"Error in model response: {str(response)}")
                continue

            if response.mode == "memory":
                curr_image_memory = response.response

                # process image memory response
                curr_image_memory = curr_image_memory.split("Description:")[-1].strip()
                if curr_image_memory.endswith("```"):
                    curr_image_memory = curr_image_memory.rstrip("```")

                # save the image content into memory
                step_info["memory"] = curr_image_memory

                # save the image content into history
                self.history_image_memory.append(curr_image_memory)
            else:
                predictions.append(response.response)

        # do majority voting
        top_actions, action_prediction_map = get_top_actions(predictions)

        # Final decision
        if top_actions:
            action = top_actions[0]
            prediction = action_prediction_map[action]
            step_info["initial_action"] = action
            step_info["thought"] = extract_thought(prediction)
        else: # Todo: return fail??
            raise ValueError("No selected actions")

        worker_end_time = time.time()
        worker_time_duration = worker_end_time - worker_start_time
        step_time_duration += worker_time_duration

        # extract the thought
        thought = extract_thought(prediction)

        # post processing: check if the current action is "click"
        if "click" in action:
            print("Running click verification...")

            # apply the xml
            get_xml(self.adb_connection)
            xml_fp = f"{self.adb_connection.get_xml_dir()}/window_dump.xml"
            xml_content = rule_based_extractor(xml_fp)
            click_check_prompt = f"""\
You are an AI agent to help me decide the next action for interacting with an Android phone. Given the thought and action type, your job is to examine from interactive actions and choose the correct action according to the action type and thought.

## Thought
{thought}

## Action Type
click

## Interactive Actions
{xml_content}

Return the answer following the output format:
Action: ...
"""
            # print(click_check_prompt)
            action_correct_start = time.time()
            action_response = self.general_sync_agent.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": click_check_prompt}],
                temperature=0.0,
            )

            action_correct_duration = time.time() - action_correct_start
            step_time_duration += action_correct_duration

            action_response = action_response.choices[0].message.content.strip()
            print(f"Corrected Action:\n{action_response}")
            try:
                # extract the coordinates from model response
                match = re.search(r"Action:\s*click,(\d+),(\d+)", action_response)
                screen_x, screen_y = match.groups()
                model_x, model_y = revert_coordinate(int(screen_x), int(screen_y))
                action = f"click(start_box='({model_x},{model_y})')"
            except:
                print(
                    "[Action verification and correction failed due to regex pattern error]"
                )

            # update the prediction
            prediction = f"Thought: {thought}\nAction: {action}"
            step_info["correct_action"] = action

        # post processing: verify the typing content
        elif "type" in action:
            print(f"Running typing content verification...")

            formatted_memory = "\n".join(
                f"  {i+1}. {item}" for i, item in enumerate(self.history_image_memory)
            )

            # check if the content
            typing_content_prompt = f"""\
You are a helpful Android mobile agent. Your job is to improve or complete a `type(content=...)` action proposed by an LLM, based on the task, its thought and the screen context.

You are given:
- A **Task**: user's input task for the mobile agent to complete.
- A **Thought**: the reasoning step from the LLM (what it is trying to do).
- An **Action**: the current `type(content=...)` command that may have errors or be incomplete.
- **Image Content Memory**: extracted information from previous phone screenshots in chronological order.

Your job:
1. **If the content has typos or incorrect information**, correct them using the memory.
2. **If the content is incomplete or vague**, enrich it using relevant phrases from the image content memory.
3. **If the task is search-related** (e.g., inputting a query), append `\\n` to the content to simulate a "submit" action.

This prompt only applies to `type(content=...)` actions.

Now, complete the output below:

Task:
{instruction}

Thought:
{thought}

Action:
{action}

Image Content Memory:
{formatted_memory}

---

Output Format:
type(content='your corrected or enriched content here')
"""
            print(f"Typing verifier template:\n{typing_content_prompt}\n")

            type_verifier_start = time.time()
            response = self.general_sync_agent.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": typing_content_prompt}],
                temperature=0.0,
            )
            type_verifier_duration = time.time() - type_verifier_start
            step_time_duration += type_verifier_duration

            try:
                action = response.choices[0].message.content.strip()
                match = re.search(r"type\(content='.*?'\)", action, re.DOTALL)
                action = match.group(0)
            except:
                print("[Typing content verification failed due to regex pattern error]")

            print(f"Typing verifier corrected action:\n{action}\n")

            # update the prediction
            prediction = f"Thought: {thought}\nAction: {action}"
            # update the action
            step_info["correct_action"] = action

        # save the thought into history
        self.history_thoughts.append(thought)
        # save the correct action into history
        self.history_actions.append(action)
        # save prediction into history
        self.history_predictions.append(add_box_token(prediction))

        # === parse and execute the action === #
        android_start_time = time.time()
        success, action_name, error_msg = self.execute_action_env(action)
        android_end_time = time.time()
        android_duration = android_end_time - android_start_time
        step_time_duration += android_duration
        # append the step time
        self.recorded_time.append(step_time_duration)

        if success and (action_name is not None):
            if action_name.startswith("finish"):
                task_end_time = time.time()
                step_info["average_step_time"] = sum(self.recorded_time) / len(self.recorded_time)
                step_info["step_time"] = step_time_duration

                self.steps_info.append(step_info)
                with open(self.log_dir / "trajectory.json", "w") as f:
                    json.dump(self.steps_info, f, indent=4)
                return base_agent.AgentInteractionResult(True, step_info)
            else:
                # wait some time for the screen to load
                # todo: remove this if the phone state is loaded through the base agent 
                # i.e. (using the _transition_pause)
                time.sleep(5)

                step_info["step_time"] = step_time_duration
                self.steps_info.append(step_info)
                with open(self.log_dir / "trajectory.json", "w") as f:
                    json.dump(self.steps_info, f, indent=4)

                # update gradio history
                new_message = f"Successfully executed {action_name} | Android execution time = {android_duration:.2f} s"
                print(new_message)

                # append the thought and action for next iteration
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": self.history_predictions[-1],
                    }
                )
                return base_agent.AgentInteractionResult(False, step_info)
        # todo: shall we still  continue or simply return error??
        elif not success and action_name is not None:
            new_message = f"Problem encountered when executing {action_name} (latency = {(android_end_time - android_start_time):.2f} s)"
        else:
            new_message = f"Exception due to {error_msg}."
        print(new_message)
        step_info["error"] = error_msg
        return base_agent.AgentInteractionResult(False, step_info)
