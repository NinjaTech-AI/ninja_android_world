import os
import time
import subprocess
from PIL import Image
from time import sleep
from pydantic import BaseModel
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv(override=True)
client = OpenAI()


class PackageInfo(BaseModel):
    package: str


class ADBConnection:
    def __init__(self, connection=None):
        self._instance = connection or os.environ.get(
            "ADB_CONNECTION", default="127.0.0.1:5555"
        )
        self._adb_path = os.environ.get("ADB_PATH", default="adb")
        self._setup_screenshot_dir()
        self._setup_xml_dir()
        print(
            f"adb connected: {self._instance} [screenshot dir: {self._screenshot_dir}, xml dir: {self._xml_dir}]"
        )

    def _setup_screenshot_dir(self):
        dir_name = self._instance.replace(".", "-").replace(":", "-")
        self._screenshot_dir = os.path.join(".", dir_name, "screenshot")
        os.makedirs(self._screenshot_dir, exist_ok=True)

    def _setup_xml_dir(self):
        dir_name = self._instance.replace(".", "-").replace(":", "-")
        self._xml_dir = os.path.join(".", dir_name, "xml")
        os.makedirs(self._xml_dir, exist_ok=True)

    def get_connection(self):
        return self._instance

    def set_connection(self, value):
        self._instance = value
        self._setup_screenshot_dir()
        self._setup_xml_dir()

    def get_screenshot_dir(self):
        return self._screenshot_dir

    def get_xml_dir(self):
        return self._xml_dir

    def get_adb_path(self):
        return self._adb_path

    def check_adb_connection(self):
        adb_path = self.get_adb_path()
        current_adb_connection = self.get_connection()
        try:
            process = subprocess.run(
                [adb_path, "devices"], capture_output=True, text=True, check=True
            )
            output = process.stdout.strip()  # Remove leading/trailing whitespace
            if not output:
                return False

            lines = output.split("\n")
            for line in lines[1:]:  # Skip the header line
                parts = line.split("\t")
                if (
                    len(parts) == 2
                    and parts[0] == current_adb_connection
                    and parts[1] == "device"
                ):
                    return True
            return False
        except subprocess.CalledProcessError as e:
            print(f"WARNING: Error running adb devices: {e}")
            return False
        except FileNotFoundError:
            print(
                "WARNING: adb command not found. Make sure adb is installed and in your PATH."
            )
            return False

    # return true if the connection is fine
    # return true if the connection is not fine, but fine after one attempt of reconnect
    def check_adb_connection_with_reconnect(self):
        if self.check_adb_connection():
            return True
        else:
            current_adb_connection = self.get_connection()
            adb_path = self.get_adb_path()
            print(
                f"WARNING: Problem with adb connection {current_adb_connection}, trying to reconnect..."
            )
            # subprocess.run([adb_path, 'disconnect', current_adb_connection], capture_output=True, text=True, check=True)
            subprocess.run(
                [adb_path, "connect", current_adb_connection],
                capture_output=True,
                text=True,
                check=False,
            )
            return self.check_adb_connection()


def get_xml(adb_connection):
    adb_path = adb_connection.get_adb_path()
    current_adb_connection = adb_connection.get_connection()
    current_xml_dir = adb_connection.get_xml_dir()

    command = f"{adb_path} -s {current_adb_connection} shell rm /sdcard/window_dump.xml"
    subprocess.run(command, capture_output=True, text=True, shell=True)
    time.sleep(0.5)

    command = f"{adb_path} -s {current_adb_connection} shell uiautomator dump /sdcard/window_dump.xml"
    subprocess.run(command, capture_output=True, text=True, shell=True)
    time.sleep(0.5)

    command = f"{adb_path} -s {current_adb_connection} pull /sdcard/window_dump.xml ./{current_xml_dir}"
    subprocess.run(command, capture_output=True, text=True, shell=True)


def get_screenshot(adb_connection):
    adb_path = adb_connection.get_adb_path()
    current_adb_connection = adb_connection.get_connection()
    current_screenshot_dir = adb_connection.get_screenshot_dir()

    command = f"{adb_path} -s {current_adb_connection} shell rm /sdcard/screenshot.png"
    subprocess.run(command, capture_output=True, text=True, shell=True)
    time.sleep(0.5)

    command = f"{adb_path} -s {current_adb_connection} shell screencap -p /sdcard/screenshot.png"
    subprocess.run(command, capture_output=True, text=True, shell=True)
    time.sleep(0.5)

    command = f"{adb_path} -s {current_adb_connection} pull /sdcard/screenshot.png ./{current_screenshot_dir}"
    subprocess.run(command, capture_output=True, text=True, shell=True)

    image_path = f"./{current_screenshot_dir}/screenshot.png"
    save_path = f"./{current_screenshot_dir}/screenshot.jpg"
    image = Image.open(image_path)
    image.convert("RGB").save(save_path, "JPEG")
    os.remove(image_path)


def start_recording(adb_connection):
    adb_path = adb_connection.get_adb_path()
    current_adb_connection = adb_connection.get_connection()

    print("Remove existing screenrecord.mp4")
    command = (
        f"{adb_path} -s {current_adb_connection} shell rm /sdcard/screenrecord.mp4"
    )
    subprocess.run(command, capture_output=True, text=True, shell=True)
    print("Start!")
    # Use subprocess.Popen to allow terminating the recording process later
    command = f"{adb_path} -s {current_adb_connection} shell screenrecord /sdcard/screenrecord.mp4"
    process = subprocess.Popen(command, shell=True)
    return process


def end_recording(adb_connection, output_recording_path):
    adb_path = adb_connection.get_adb_path()
    current_adb_connection = adb_connection.get_connection()

    print("Stopping recording...")
    # Send SIGINT to stop the screenrecord process gracefully
    stop_command = (
        f"{adb_path} -s {current_adb_connection} shell pkill -SIGINT screenrecord"
    )
    subprocess.run(stop_command, capture_output=True, text=True, shell=True)
    sleep(1)  # Allow some time to ensure the recording is stopped

    print("Pulling recorded file from device...")
    pull_command = f"{adb_path} -s {current_adb_connection} pull /sdcard/screenrecord.mp4 {output_recording_path}"
    subprocess.run(pull_command, capture_output=True, text=True, shell=True)
    print(f"Recording saved to {output_recording_path}")


def save_screenshot_to_file(adb_connection, file_path="screenshot.png"):
    """
    Captures a screenshot from an Android device using ADB, saves it locally, and removes the screenshot from the device.

    Args:
        adb_path (str): The path to the adb executable.

    Returns:
        str: The path to the saved screenshot, or raises an exception on failure.
    """
    # Define the local filename for the screenshot
    local_file = file_path
    adb_path = adb_connection.get_adb_path()
    current_adb_connection = adb_connection.get_connection()

    if os.path.dirname(local_file) != "":
        os.makedirs(os.path.dirname(local_file), exist_ok=True)

    # Define the temporary file path on the Android device
    device_file = "/sdcard/screenshot.png"

    try:
        # print("\tRemoving existing screenshot from the Android device...")
        command = (
            f"{adb_path} -s {current_adb_connection} shell rm /sdcard/screenshot.png"
        )
        subprocess.run(command, capture_output=True, text=True, shell=True)
        time.sleep(0.5)

        # Capture the screenshot on the device
        # print("\tCapturing screenshot on the Android device...")
        result = subprocess.run(
            f"{adb_path} -s {current_adb_connection} shell screencap -p {device_file}",
            capture_output=True,
            text=True,
            shell=True,
        )
        time.sleep(0.5)
        if result.returncode != 0:
            raise RuntimeError(
                f"Error: Failed to capture screenshot on the device. {result.stderr}"
            )

        # Pull the screenshot to the local computer
        # print("\tTransferring screenshot to local computer...")
        result = subprocess.run(
            f"{adb_path} -s {current_adb_connection} pull {device_file} {local_file}",
            capture_output=True,
            text=True,
            shell=True,
        )
        time.sleep(0.5)
        if result.returncode != 0:
            raise RuntimeError(
                f"Error: Failed to transfer screenshot to local computer. {result.stderr}"
            )

        # Remove the screenshot from the device
        # print("\tRemoving screenshot from the Android device...")
        result = subprocess.run(
            f"{adb_path} -s {current_adb_connection} shell rm {device_file}",
            capture_output=True,
            text=True,
            shell=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Error: Failed to remove screenshot from the device. {result.stderr}"
            )

        # print(f"\tAtomic Operation Screenshot saved to {local_file}")
        return local_file

    except Exception as e:
        print(str(e))
        return None


def tap(adb_connection, x, y):
    adb_path = adb_connection.get_adb_path()
    current_adb_connection = adb_connection.get_connection()
    command = f"{adb_path} -s {current_adb_connection} shell input tap {x} {y}"
    subprocess.run(command, capture_output=True, text=True, shell=True)


def type(adb_connection, text):
    adb_path = adb_connection.get_adb_path()
    current_adb_connection = adb_connection.get_connection()

    text = text.replace("\\n", "_").replace("\n", "_")
    for char in text:
        if char == " ":
            command = f"{adb_path} -s {current_adb_connection} shell input text %s"
            subprocess.run(command, capture_output=True, text=True, shell=True)
        elif char == "_":
            command = f"{adb_path} -s {current_adb_connection} shell input keyevent 66"
            subprocess.run(command, capture_output=True, text=True, shell=True)
        elif "a" <= char <= "z" or "A" <= char <= "Z" or char.isdigit():
            command = f"{adb_path} -s {current_adb_connection} shell input text {char}"
            subprocess.run(command, capture_output=True, text=True, shell=True)
        elif char in "-.,!?@'°/:;()":
            command = (
                f'{adb_path} -s {current_adb_connection} shell input text "{char}"'
            )
            subprocess.run(command, capture_output=True, text=True, shell=True)
        else:
            command = f'{adb_path} -s {current_adb_connection} shell am broadcast -a ADB_INPUT_TEXT --es msg "{char}"'
            subprocess.run(command, capture_output=True, text=True, shell=True)


def enter(adb_connection):
    adb_path = adb_connection.get_adb_path()
    current_adb_connection = adb_connection.get_connection()
    command = (
        f"{adb_path} -s {current_adb_connection} shell input keyevent KEYCODE_ENTER"
    )
    subprocess.run(command, capture_output=True, text=True, shell=True)


def backspace(adb_connection):
    adb_path = adb_connection.get_adb_path()
    current_adb_connection = adb_connection.get_connection()
    command = f"{adb_path} -s {current_adb_connection} shell input keyevent 67"
    subprocess.run(command, capture_output=True, text=True, shell=True)


def delete(adb_connection):
    adb_path = adb_connection.get_adb_path()
    current_adb_connection = adb_connection.get_connection()
    command = f"{adb_path} -s {current_adb_connection} shell input keyevent 112"
    subprocess.run(command, capture_output=True, text=True, shell=True)


def volume_up(adb_connection):
    adb_path = adb_connection.get_adb_path()
    current_adb_connection = adb_connection.get_connection()
    command = (
        f"{adb_path} -s {current_adb_connection} shell input keyevent KEYCODE_VOLUME_UP"
    )
    subprocess.run(command, capture_output=True, text=True, shell=True)


def volume_down(adb_connection):
    adb_path = adb_connection.get_adb_path()
    current_adb_connection = adb_connection.get_connection()
    command = f"{adb_path} -s {current_adb_connection} shell input keyevent KEYCODE_VOLUME_DOWN"
    subprocess.run(command, capture_output=True, text=True, shell=True)


def swipe(adb_connection, x1, y1, x2, y2, time=500):
    adb_path = adb_connection.get_adb_path()
    current_adb_connection = adb_connection.get_connection()
    command = f"{adb_path} -s {current_adb_connection} shell input swipe {x1} {y1} {x2} {y2} {time}"
    subprocess.run(command, capture_output=True, text=True, shell=True)


def back(adb_connection):
    adb_path = adb_connection.get_adb_path()
    current_adb_connection = adb_connection.get_connection()
    command = f"{adb_path} -s {current_adb_connection} shell input keyevent 4"
    subprocess.run(command, capture_output=True, text=True, shell=True)


def home(adb_connection):
    adb_path = adb_connection.get_adb_path()
    current_adb_connection = adb_connection.get_connection()
    # command = adb_path + f" shell am start -a android.intent.action.MAIN -c android.intent.category.HOME"
    command = (
        f"{adb_path} -s {current_adb_connection} shell input keyevent KEYCODE_HOME"
    )
    subprocess.run(command, capture_output=True, text=True, shell=True)


def switch_app(adb_connection):
    adb_path = adb_connection.get_adb_path()
    current_adb_connection = adb_connection.get_connection()
    command = f"{adb_path} -s {current_adb_connection} shell input keyevent KEYCODE_APP_SWITCH"
    subprocess.run(command, capture_output=True, text=True, shell=True)


def open_app_with_name(adb_connection: ADBConnection, app_name: str) -> bool:
    """
    Opens an app using ADB given its common name. First tries to find package via ADB,
    falls back to hardcoded values if no unique match is found. App name matching is case insensitive.
    1. If the app is already opened in the foreground, there is no impact
    2. If the app is not in the foreground (including the case where it is in the background),
      the app will be opened or brought to the foreground,

    Args:
        adb_connection (ADBConnection): Instance of ADBConnection class
        app_name (str): Common name of the app (e.g., 'linkedin', 'youtube')

    Returns:
        bool: True if app was successfully opened, False otherwise

    Raises:
        RuntimeError: If there's an error listing packages or executing ADB commands
    """

    # get all installed packages
    try:
        command = [
            adb_connection.get_adb_path(),
            "-s",
            adb_connection.get_connection(),
            "shell",
            "pm",
            "list",
            "packages",
        ]
        process = subprocess.run(command, capture_output=True, text=True, check=True)
        packages = process.stdout.strip().split("\n")

        formatted_packages = "\n".join(f"- {item}" for item in packages)

        system_prompt = f"You are a helpful Android expert. You are given a list of android installed packages:\n{formatted_packages}"
        user_prompt = f"I want to open the android app: {app_name}, return the correct package name from the given list."

        completion = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=PackageInfo,
        )
        message = completion.choices[0].message
        assert message.parsed, "Error when parsing the package name"
        package_name = message.parsed.package

        # Launch the app using monkey command
        command = [
            adb_connection.get_adb_path(),
            "-s",
            adb_connection.get_connection(),
            "shell",
            "monkey",
            "-p",
            package_name,
            "-c",
            "android.intent.category.LAUNCHER",
            "1",
        ]

        process = subprocess.run(command, capture_output=True, text=True, check=True)

        if process.returncode == 0:
            print(f"Successfully launched {app_name}")
        else:
            error_msg = f"Failed to launch {app_name}: {process.stderr}"
            print(error_msg)
            raise RuntimeError(error_msg)

    except Exception as e:
        # Re-raise any other exceptions that might occur
        print(f"Unexpected error: {e}")
        raise


## todo: the following control does not work yet
def close_app_with_name(adb_connection: ADBConnection, app_name: str) -> bool:
    """
    Closes an app using ADB given its common name using force-stop command.

    Args:
        adb_connection (ADBConnection): Instance of ADBConnection class
        app_name (str): Common name of the app (e.g., 'linkedin', 'youtube')

    Returns:
        bool: True if app was successfully closed, False otherwise
    """
    # Fallback dictionary for common apps (all lowercase keys)
    app_packages = {
        "linkedin": "com.linkedin.android",
        "youtube": "com.google.android.youtube",
        "music": "com.google.android.apps.youtube.music",
        "booking": "com.booking",
        "chrome": "com.android.chrome",
        "gmail": "com.google.android.gm",
    }

    def find_matching_packages(search_term: str) -> list:
        """Helper function to find packages matching a search term using ADB"""
        try:
            command = [
                adb_connection.get_adb_path(),
                "-s",
                adb_connection.get_connection(),
                "shell",
                "pm",
                "list",
                "packages",
            ]
            process = subprocess.run(
                command, capture_output=True, text=True, check=True
            )
            packages = process.stdout.strip().split("\n")

            matches = []
            search_term = search_term.lower()
            for pkg in packages:
                pkg_name = pkg.replace("package:", "").strip()
                if search_term in pkg_name.lower():
                    matches.append(pkg_name)
            return matches

        except subprocess.CalledProcessError as e:
            print(f"Error listing packages: {e}")
            return []

    # Convert app_name to lowercase for all comparisons
    app_name = app_name.lower()

    # Try to find package via ADB first
    matching_packages = find_matching_packages(app_name)

    package_name = None
    if len(matching_packages) == 1:
        package_name = matching_packages[0]
    elif len(matching_packages) > 1:
        print(f"Warning: Multiple matches found for '{app_name}': {matching_packages}")
        print("Using fallback package dictionary instead")
        package_name = app_packages.get(app_name)

    if not package_name:
        package_name = app_packages.get(app_name)
        if not package_name:
            print(f"Error: Could not find package for '{app_name}'")
            if matching_packages:
                print("Matches found:", matching_packages)
            return False

    try:
        # Force stop the app
        force_stop_command = [
            adb_connection.get_adb_path(),
            "-s",
            adb_connection.get_connection(),
            "shell",
            "am",
            "force-stop",
            package_name,
        ]

        subprocess.run(force_stop_command, capture_output=True, text=True, check=True)
        print(f"Successfully closed {app_name} (package: {package_name})")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Error closing {app_name}: {e}")
        return False


def close_all_apps(adb_connection: ADBConnection) -> bool:
    """
    Closes all running apps using ADB force-stop command.

    Args:
        adb_connection (ADBConnection): Instance of ADBConnection class

    Returns:
        bool: True if operation was successful, False otherwise
    """
    try:
        # Get list of running apps
        running_apps_command = [
            adb_connection.get_adb_path(),
            "-s",
            adb_connection.get_connection(),
            "shell",
            'ps | grep -i "u0_a"',  # This finds user apps
        ]

        process = subprocess.run(running_apps_command, capture_output=True, text=True)
        running_apps = process.stdout.strip().split("\n")

        success = True
        processed_packages = set()

        for app in running_apps:
            if not app:  # Skip empty lines
                continue

            try:
                # Extract package name from ps output
                parts = app.split()
                if len(parts) >= 8:  # ps output typically has 8 or more columns
                    package_name = parts[
                        -1
                    ]  # Last column usually contains package/process name

                    # Skip if we've already processed this package
                    if package_name in processed_packages:
                        continue

                    # Skip system packages
                    if package_name.startswith(
                        "com.android"
                    ) or package_name.startswith("com.google.android"):
                        continue

                    processed_packages.add(package_name)

                    # Force stop the app
                    force_stop_command = [
                        adb_connection.get_adb_path(),
                        "-s",
                        adb_connection.get_connection(),
                        "shell",
                        "am",
                        "force-stop",
                        package_name,
                    ]

                    subprocess.run(
                        force_stop_command, capture_output=True, text=True, check=True
                    )
                    print(f"Closed {package_name}")

            except subprocess.CalledProcessError as e:
                print(f"Error closing {package_name}: {e}")
                success = False

        if success:
            print("Successfully closed all running apps")
        else:
            print("Some apps may not have been closed properly")

        return success

    except subprocess.CalledProcessError as e:
        print(f"Error during operation: {e}")
        return False
