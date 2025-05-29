"""Ninja Android agent, fast scaffold based on UI-TARS"""

from dotenv import load_dotenv

load_dotenv(override=True)

import asyncio
import base64
import json
import io
import os
import re
import sys
import time
import numpy as np

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from absl import logging

from android_world.agents import base_agent
from android_world.env import interface
from android_world.env import adb_utils
from android_world.env import representation_utils

from openai import AsyncOpenAI, OpenAI
from PIL import Image

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
            logging.info(f'Ninja Agent: screenshot_log_path={self.screenshot_log_path}')
            self.screenshot_log_path.mkdir(parents=True, exist_ok=True)

            # Set screen dimensions
            self.WIDTH, self.HEIGHT = self.env.logical_screen_size
    
    def _extract_clickable_elements(self, ui_elements: list[representation_utils.UIElement]) -> str:
        """Extract clickable elements from list of UIElements.
        
        Args:
            ui_elements: List of UIElement objects from environment state
            
        Returns:
            str: Formatted string containing clickable elements and their actions
        """

        # Get UI elements from environment if not provided or empty
        if not ui_elements:
            ui_elements = self.env.get_state(wait_to_stabilize=False).ui_elements

        output = ""
        current_id = 1
        
        output += "=== CLICKABLE ELEMENTS ===\n"
        
        for element in ui_elements:
            if element.is_clickable:
                # Get display text (prioritize different text sources)
                display_text = ""
                if element.text:
                    display_text = element.text
                elif element.content_description:
                    display_text = element.content_description
                elif element.resource_id:
                    # Clean up resource ID similar to original function
                    id_parts = element.resource_id.split('/')
                    display_text = id_parts[-1] if len(id_parts) > 1 else element.resource_id
                    
                if display_text:
                    display_text = f"\"{display_text}\""
                else:
                    continue  # Skip elements without any text
                    
                # Get class name
                class_name = element.class_name if element.class_name else "android.view.View"
                
                # Get bounds
                if element.bbox_pixels:
                    bounds = f"[{int(element.bbox_pixels.x_min)},{int(element.bbox_pixels.y_min)}][{int(element.bbox_pixels.x_max)},{int(element.bbox_pixels.y_max)}]"
                    
                    # Calculate middle point for action
                    middle_x = int((element.bbox_pixels.x_min + element.bbox_pixels.x_max) // 2)
                    middle_y = int((element.bbox_pixels.y_min + element.bbox_pixels.y_max) // 2)
                    
                    # Format output similar to original
                    output += f"{display_text} ({class_name})\n"
                    output += f"Bounds: {bounds}\n"
                    output += f"{current_id}. Action: click,{middle_x},{middle_y}\n"
                    output += "\n"
                    
                    current_id += 1
                    
        return output

    def _revert_coordinate(self, screen_x: int, screen_y: int):
        model_x = int(screen_x * 1000 / self.WIDTH)
        model_y = int(screen_y * 1000 / self.HEIGHT)
        return model_x, model_y

    def _execute_action_env(self, action: str) -> tuple[bool, str | None, str | None]:
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

            if "click" in action:
                coord_match = re.search(r"start_box='\((\d+),(\d+)\)'", action)
                if coord_match:
                    x = int(coord_match.group(1))
                    y = int(coord_match.group(2))
                    
                    # Normalize coordinates from 1000-based to actual screen dimensions
                    normalized_x = int(x * self.WIDTH / 1000)
                    normalized_y = int(y * self.HEIGHT / 1000)
                    
                    adb_utils.tap_screen(normalized_x, normalized_y, self.env.controller)
                    return True, f"tap({normalized_x}, {normalized_y})", None

            elif "type(content" in action:  # Note: using Crane's original implementation
                def escape_quotes(match):
                    content = match.group(1)  # get content value
                    return content

                pattern = r"type\(content='(.*?)'\)"  # match type(content='...')
                content = re.sub(pattern, escape_quotes, action)

                # process string
                text = escape_single_quotes(content)
                logging.info("raw text for typing is:", repr(text))

                action_name = f"type({text})"
                adb_utils.type_text(text, self.env.controller)

                if text.endswith("\n"):
                    adb_utils.press_enter_button(self.env.controller)

                return True, action_name, None

            elif "scroll" in action:
                start_coord_match = re.search(r"start_box='\((\d+),(\d+)\)'", action)
                end_coord_match = re.search(r"end_box='\((\d+),(\d+)\)'", action)
                if start_coord_match and end_coord_match:
                    x1 = int(start_coord_match.group(1))
                    y1 = int(start_coord_match.group(2))
                    x2 = int(end_coord_match.group(1))
                    y2 = int(end_coord_match.group(2))
                    
                    # Normalize coordinates
                    normalized_x1 = int(x1 * self.WIDTH / 1000)
                    normalized_y1 = int(y1 * self.HEIGHT / 1000)
                    normalized_x2 = int(x2 * self.WIDTH / 1000)
                    normalized_y2 = int(y2 * self.HEIGHT / 1000)
                    
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
                    normalized_x = int(x * self.WIDTH / 1000)
                    normalized_y = int(y * self.HEIGHT / 1000)
                    
                    adb_utils.long_press(normalized_x, normalized_y, self.env.controller)
                    return True, f"long_press({normalized_x}, {normalized_y})", None

            elif "wait" in action:
                time.sleep(2.0)
                return True, "wait()", None

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
            "step": 0.0,
        }

        # Todo, use the base_agent's supporting function to retrieve the phone state
        # where is also handle the transition consistently??
        state = self.env.get_state(wait_to_stabilize=False)
        # Get screenshot and encode
        screenshot = state.pixels.copy()
        base64_image = base64_encode_image(screenshot)
        step = len(self.steps_info)
        step_info['step'] = step

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
                logging.warning(f"Error in model response: {str(response)}")
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
#            logging.warning("Running click verification...")

            clickable_elements = self._extract_clickable_elements(state.ui_elements)
            click_check_prompt = f"""\
You are an AI agent to help me decide the next action for interacting with an Android phone. Given the thought and action type, your job is to examine from interactive actions and choose the correct action according to the action type and thought.

## Thought
{thought}

## Action Type
click

## Interactive Actions
{clickable_elements}

Return the answer following the output format:
 ...
"""
            # logging.warning(click_check_prompt)
            action_correct_start = time.time()
            action_response = self.general_sync_agent.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": click_check_prompt}],
                temperature=0.0,
            )

            action_correct_duration = time.time() - action_correct_start
            step_time_duration += action_correct_duration

            action_response = action_response.choices[0].message.content.strip()
            logging.warning(f"Corrected Action: {action_response}")
            try:
                # extract the coordinates from model response
                match = re.search(r"Action:\s*click,(\d+),(\d+)", action_response)
                screen_x, screen_y = match.groups()
                model_x, model_y = self._revert_coordinate(int(screen_x), int(screen_y))
                action = f"click(start_box='({model_x},{model_y})')"
            except:
                logging.warning(
                    "[Action verification and correction failed due to regex pattern error]"
                )

            # update the prediction
            prediction = f"Thought: {thought}\nAction: {action}"
            step_info["correct_action"] = action

        # post processing: verify the typing content
        elif "type" in action:
            # logging.warning(f"Running typing content verification...")

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
            logging.warning(f"Typing verifier template:\n{typing_content_prompt}\n")

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
                logging.warning("[Typing content verification failed due to regex pattern error]")

            logging.warning(f"Typing verifier corrected action: {action}")

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
        success, action_name, error_msg = self._execute_action_env(action)
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
                logging.warning(new_message)

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
        logging.warning(new_message)
        step_info["error"] = error_msg
        return base_agent.AgentInteractionResult(False, step_info)
