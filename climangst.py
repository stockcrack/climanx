import random
import streamlit as st
from openai import OpenAI
import yaml
from PIL import Image
from io import BytesIO
import requests
from dotenv import load_dotenv
import os
import logging


load_dotenv()

LOGGING_LEVELS = {"debug": logging.DEBUG, "info": logging.INFO,
                  "warning": logging.WARNING, "error": logging.ERROR, "critical": logging.CRITICAL}


# Load API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")

STABILITY_ENDPOINTS = {
    "Stable Diffusion 3": "https://api.stability.ai/v2beta/stable-image/generate/sd3",
    "Stable Image Core": "https://api.stability.ai/v2beta/stable-image/generate/core",
    "Stable Image Ultra": "https://api.stability.ai/v2beta/stable-image/generate/ultra",
}

# Directory where YAML files are stored
questions_directory = "questions"


# Function to load questions and prompts from a YAML file
def load_questions(yaml_file: str) -> dict:
    with open(yaml_file, "r") as file:
        return yaml.safe_load(file)


def chat_completion(user_prompt: str, system_prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )
    return response.choices[0].message.content.strip()


def ensure_prompt(prompt: str, max_length: int = 1024) -> str:
    """
    Ensure the prompt is less than or equal to max_length characters.
    If the prompt exceeds the max_length, ask OpenAI to summarize it.

    Parameters:
    - prompt (str): The original prompt to be checked and potentially shortened.
    - max_length (int): The maximum allowed length for the prompt (default is 1024 characters).

    Returns:
    - str: The original prompt if within the limit, otherwise a summarized version.
    """
    if len(prompt) <= max_length:
        return prompt

    # If the prompt is too long, request a summary
    summarized_prompt = chat_completion(system_prompt="You are a helpful assistant who accurately summarizes long prompts for generating images.",
                                        user_prompt=f"The following text is too long for the model to process. Please summarize it in a concise and clear manner, ensuring it retains the key details:\n\n{prompt}")

    # Ensure the summarized prompt is still within the max_length limit
    if len(summarized_prompt) > max_length:
        raise ValueError(
            "The summarized prompt is still too long. Try adjusting the summary parameters."
        )
    logging.info(
        f"Shortened prompt from {len(prompt)} to {len(summarized_prompt)} characters.")
    logging.info(f"Original prompt: {prompt}")
    logging.info(f"Summarized prompt: {summarized_prompt}")
    return summarized_prompt


# MAGIC_PROMPT = "I NEED to test how the tool works with extremely simple prompts. DO NOT add any detail, just use it AS-IS: "
MAGIC_PROMPT = ""


def generate_image(prompt: str) -> Image:
    if st.session_state.ai == "DALL-E 3":
        return generate_image_dalle3(prompt)
    elif st.session_state.ai.startswith("Stable"):
        return generate_image_stability(prompt)
    else:
        st.sidebar.write("Please select an AI model.")
        return None


def generate_image_stability(prompt: str) -> Image:
    new_prompt = ensure_prompt(prompt, max_length=10000)
    endpoint = STABILITY_ENDPOINTS[st.session_state.ai]
    headers = {"Authorization": f"Bearer {STABILITY_API_KEY}",
               "Accept": "image/*"}
    files = {"none": ""}
    data = {"prompt": new_prompt, "output_format": "jpeg"}

    try:
        response = requests.post(
            endpoint,
            headers=headers,
            files=files,
            data=data,
        )

        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        else:
            raise Exception(str(response.json()))
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None


# Function to generate or modify an image with DALL-E
def generate_image_dalle3(prompt: str) -> Image:
    full_prompt = MAGIC_PROMPT + prompt
    new_prompt = ensure_prompt(full_prompt, max_length=1000)
    try:
        response = client.images.generate(
            prompt=new_prompt, n=1, size="1024x1024", model="dall-e-3"
        )
        image_url = response.data[0].url
        return Image.open(BytesIO(requests.get(image_url).content))
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None


# Get the list of YAML files from the "questions" subdirectory
def get_yaml_files(directory: str) -> list[str]:
    return [f for f in os.listdir(directory) if f.endswith(".yaml")]


def select_image(emo, prompt, image, response):
    st.session_state.current_image = image
    st.session_state.image_history.append(response["prompt"])
    st.session_state.responses.append(response)
    logging.info(f"Selected image for emotion: {emo}")


def save_image(image: Image, filename: str = "climangst_image.png"):
    # Allow the user to download the image
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    st.download_button(
        label="Download Final Image",
        data=buffer,
        file_name=filename,
        mime="image/png",
    )


def read_strings_from_file(file_path: str) -> list[str]:
    with open(file_path, "r") as file:
        return [line.strip() for line in file.readlines()]


def start_over():
    st.session_state.current_state = "unintialized"


def start_questionnaire():
    st.session_state.current_state = "start_questionnaire"


def initialize():
    if "logging_configured" not in st.session_state:
        if "VLINDGUIDE_LOGGING_LEVEL" not in os.environ:
            os.environ["VLINDGUIDE_LOGGING_LEVEL"] = "info"
        logging.basicConfig(
            # Set the logging level
            level=LOGGING_LEVELS[os.getenv(
                "VLINDGUIDE_LOGGING_LEVEL").lower()],
            # Format for log messages
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler("vlindguide.log"),
                      logging.StreamHandler()]      # Log to console
        )
        st.session_state.logging_configured = True
        logging.info("initialize(): Logging configured.")

    logging.info("initialize(): Initializing the VlindGuide application.")

    st.sidebar.write(
        "Hi! I'm Max Vlinder, your climate guide. I'll be with you throughout this experience.   First, let's set up your journey."
    )
    # Choose the AI
    ai_choices = ["DALL-E 3"] + list(STABILITY_ENDPOINTS.keys())
    st.session_state.ai = st.sidebar.selectbox(
        "Choose an AI model:", ai_choices, 0)
    logging.info(f"Selected AI model: {st.session_state.ai}")

    # Get the list of YAML files
    yaml_files = [os.path.splitext(filename)[0]
                  for filename in get_yaml_files(questions_directory)]
    selected_yaml = st.sidebar.selectbox(
        "Choose a setting:", yaml_files, 0)
    # Load questions and prompts based on the selected YAML file
    logging.info(f"Selected YAML file: {selected_yaml}")
    st.session_state.yaml_path = os.path.join(
        questions_directory, selected_yaml + ".yaml")
    st.session_state.question_data = load_questions(st.session_state.yaml_path)

    if not st.session_state.question_data:
        st.error("Error loading questions from the selected YAML file.")
        st.stop()

    drawing_styles = read_strings_from_file("drawing_styles.txt")
    # initial_choice = random.randint(0, len(drawing_styles) - 1)
    initial_choice = 0
    st.session_state.drawing_style = st.sidebar.selectbox(
        "Choose an artistic style:",
        drawing_styles,
        initial_choice,
    ).lower()

    logging.info(f"Selected drawing style: {st.session_state.drawing_style}")

    st.sidebar.button("Start", on_click=start_questionnaire)


def start_questionnaire():
    logging.info("start_questionnaire(): Starting the questionnaire.")
    st.session_state.current_question = 0
    drawing_style = st.session_state.drawing_style
    question_data = st.session_state.question_data
    yaml_path = st.session_state.yaml_path
    st.session_state.initial_prompt = (
        f'{question_data["initial_image_prompt"]}'
    )
    st.session_state.image_history = [st.session_state.initial_prompt]
    st.session_state.responses = []
    st.session_state.question_data = question_data
    st.session_state.yaml_path = yaml_path
    st.session_state.questions = question_data["questions"]
    logging.info(f"Loaded {len(st.session_state.questions)} questions.")
    logging.info(f"Initial prompt: {st.session_state.initial_prompt}")

    with st.spinner(f"Generating initial image with {st.session_state.ai}..."):
        st.sidebar.write("I am creating an initial image for you to ponder...")
        generated_image = generate_image(
            f"{st.session_state.initial_prompt}  The viewer should feel relaxed.  Use artistic style {st.session_state.drawing_style}.")
        if generated_image is not None:
            st.session_state.current_image = generated_image
            logging.info("Initial image generated successfully.")
        else:
            logging.error("Error generating initial image.")
            st.error("An error occurred while generating the image.")
            st.stop()

    st.sidebar.write("Here is an image to contemplate to start your journey.")
    st.sidebar.write(
        "Let's begin!  After every question, select the emotion and image that resonates with you the most."
    )
    st.session_state.current_state = "question"


def next_question():
    # Display next question
    logging.info(
        f"next_question(): Displaying question {st.session_state.current_question}.")
    st.image(st.session_state.current_image, caption="Your current image")
    question = st.session_state.questions[st.session_state.current_question]
    st.write(
        f"## {st.session_state.current_question+1}: **{question['question']}**")

    # Show options as small images
    ncols = len(question["responses"])
    base_prompt = " ".join(st.session_state.image_history)
    cols = st.columns(ncols)
    for i, response in enumerate(question["responses"]):
        with cols[i]:
            generated_image = None
            emo = response["emotion"]

            prompt = (
                base_prompt
                + " "
                + response["prompt"]
                + " "
                + f"The viewer should feel {response['emotion'].lower()}.  Use artistic style {st.session_state.drawing_style}."
            )
            logging.info(f"Generating image for {emo} with prompt: \n{prompt}")
            with st.spinner(
                f"Creating image for {response['emotion'].lower()}..."
            ):
                generated_image = generate_image(prompt)
            if generated_image is not None:
                st.button(
                    emo,
                    on_click=select_image,
                    args=(emo, prompt, generated_image, response),
                    type="primary"
                )
                st.image(generated_image, caption=response["emotion"])

            else:
                st.error("An error occurred while generating the image.")
                logging.error("Error generating image for question.")
    st.session_state.current_question += 1
    logging.info(
        f"next_question(): Incremented current question to {st.session_state.current_question} out of {len(st.session_state.questions)}.")
    if (st.session_state.current_question == len(st.session_state.questions)):
        st.session_state.current_state = "finalize"


def finalize():
    # Show final image and summary
    logging.info("finalize(): Displaying final image and summary.")
    st.sidebar.write(
        "Thank you for completing our journey together! Here is the final image representing your journey."
    )

    if st.session_state.current_image is not None:
        st.image(
            st.session_state.current_image, caption="Your Climate Anxiety Reflection"
        )

    # Show summary of emotions
    journey = "Here is a summary of your emotional journey through the climate anxiety questions:\n\n"
    st.sidebar.write("### Your Emotional Journey")
    for i, (question, response) in enumerate(
        zip(st.session_state.questions, st.session_state.responses)
    ):
        qtext = f"**Question {i+1}**: {question['question']} => {response['emotion']}"
        st.sidebar.write(qtext)
        journey += qtext + "\n"

    profile = chat_completion(
        user_prompt=journey, system_prompt="Please summarize my emotional journey through the climate anxiety questions.  Find deeper psychological insights into my personality.  Identify a relevant psychological profile and recommend a course of action."
    )
    logging.info(f"Generated profile: {profile}")
    st.sidebar.write("### Emotional Profile")
    st.sidebar.write(profile)

    # Save image option
    st.button("Save Image", on_click=save_image,
              args=(st.session_state.current_image,))
    st.button("Start Over", on_click=start_over)

    # Option to email the image (for future implementation)
    # email = st.text_input("Enter your email to receive the final image and summary:")
    # if st.button("Send Email"):
    #     send_email(email, final_image, st.session_state.responses)


def main():
    logging.info("main(): Starting the VlindGuide application.")

    if "current_state" not in st.session_state:
        logging.info("main(): Initializing session state.")
        st.session_state.current_state = "unintialized"

    # Display Butterfly Assistant
    st.sidebar.image("butterfly.png", use_column_width=True)

    # Set up the UI for YAML file selection
    st.title("VlindGuide")
    st.subheader("Explore your feelings about climate change")

    # Initialize session state
    if st.session_state.current_state == "unintialized":
        initialize()
    elif st.session_state.current_state == "start_questionnaire":
        start_questionnaire()
    elif st.session_state.current_state == "question":
        next_question()
    elif st.session_state.current_state == "finalize":
        finalize()
    else:
        logging.error(f"Unknown state: {st.session_state.current_state}")
        st.error(f"Unknown state: {st.session_state.current_state}")
        st.stop()


main()
