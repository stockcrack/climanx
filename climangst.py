import streamlit as st
from openai import OpenAI
import yaml
from PIL import Image
from io import BytesIO
import requests
from dotenv import load_dotenv
import os
import typing

load_dotenv()

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
def load_questions(yaml_file):
    with open(yaml_file, "r") as file:
        return yaml.safe_load(file)


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
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant who accurately summarizes long prompts for generating images.",
            },
            {
                "role": "user",
                "content": f"The following text is too long for the model to process. Please summarize it in a concise and clear manner, ensuring it retains the key details:\n\n{prompt}",
            },
        ],
    )

    # Extract the summary from the response
    summarized_prompt = response.choices[0].message.content.strip()

    # Ensure the summarized prompt is still within the max_length limit
    if len(summarized_prompt) > max_length:
        raise ValueError(
            "The summarized prompt is still too long. Try adjusting the summary parameters."
        )

    return summarized_prompt


# MAGIC_PROMPT = "I NEED to test how the tool works with extremely simple prompts. DO NOT add any detail, just use it AS-IS: "
MAGIC_PROMPT = ""


def generate_image(prompt_chain):
    if st.session_state.ai == "DALL-E 3":
        return generate_image_dalle3(prompt_chain)
    elif st.session_state.ai.startswith("Stable"):
        return generate_image_stability(prompt_chain)
    else:
        st.sidebar.write("Please select an AI model.")
        return None


def generate_image_stability(prompt_chain):
    full_prompt = MAGIC_PROMPT + " ".join(prompt_chain)
    new_prompt = ensure_prompt(full_prompt, max_length=10000)
    if new_prompt != full_prompt:
        st.sidebar.write(
            f"The prompt was too long and has been shortened to the following: {new_prompt}"
        )
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
def generate_image_dalle3(prompt_chain):
    full_prompt = MAGIC_PROMPT + " ".join(prompt_chain)
    new_prompt = ensure_prompt(full_prompt, max_length=1000)
    if new_prompt != full_prompt:
        st.sidebar.write(
            f"The prompt was too long and has been shortened to the following: {new_prompt}"
        )
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
def get_yaml_files(directory):
    return [f for f in os.listdir(directory) if f.endswith(".yaml")]


def select_image(emo, prompt, image, response):
    st.session_state.current_image = image
    st.session_state.image_history.append(response["prompt"])
    st.session_state.responses.append(response)
    st.sidebar.write(f"You selected: {emo}, and I have updated the image.")
    st.session_state.current_question += 1


def save_image(image, filename="climangst_image.png"):
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


def start_over():
    st.session_state.current_question = -1


def set_show_prompts(value):
    st.session_state.show_prompts = value


# Display Butterfly Assistant
st.sidebar.image("butterfly.png", use_column_width=True)
st.sidebar.write(
    "Hi! I'm Max Vlinder, your climate guide. I'll be with you throughout this experience."
)

# Set up the UI for YAML file selection
st.title("VlindGuide")
st.subheader("Explore your feelings about climate change")

# Initialize session state
if (
    "current_question" not in st.session_state
    or st.session_state.current_question == -1
):
    # Choose the AI
    ai_choices = list(STABILITY_ENDPOINTS.keys()) + ["DALL-E 3"]

    selected_ai = st.sidebar.selectbox(
        "Choose an AI model:",
        ai_choices,
        None,
    )
    if not selected_ai:
        st.sidebar.write("Please select an AI model.")
        st.stop()

    st.session_state.ai = selected_ai

    # Get the list of YAML files
    yaml_files = get_yaml_files(questions_directory)
    selected_yaml = st.sidebar.selectbox(
        "Choose a questionnaire:", yaml_files, None)

    # Load questions and prompts based on the selected YAML file
    if not selected_yaml:
        st.sidebar.write("Please select a questionnaire.")
        st.stop()

    yaml_path = os.path.join(questions_directory, selected_yaml)
    question_data = load_questions(yaml_path)

    if not question_data:
        st.error("Error loading questions from the selected YAML file.")
        st.stop()

    drawing_style = st.sidebar.selectbox(
        "Choose a drawing style:",
        [
            "Photorealistic",
            "Cartoon",
            "Abstract",
            "Expressionist",
            "Cubist",
            "Impressionist",
            "Naturalist",
            "Ukiyo-e",
            "Shui-mo hua"
            "Medieval",
            "Futurist",
            "Fantasy",
            "Fauvist",
            "Renaissance",
            "Baroque",
            "Neoclassical",
            "Romantic",
            "Surrealist",
            "Street Art",
            "Comic Strip",
            "Aboriginal Art",
            "Abstract Expressionist",
            "Tibetan Thangka",
            "Pop Art",
            "Op Art",
            "Pointillist",
            "Minimalist",
            "Art Nouveau",
            "Art Deco",
            "Maori Kowhaiwhai",
            "Korean Minhwa",
            "Persian Miniature",
            "Indian Mugham",
        ],
        None,
    )
    if not drawing_style:
        st.sidebar.write("Please select a drawing style.")
        st.stop()

    st.session_state.drawing_style = drawing_style.lower()

    st.session_state.show_prompts = st.sidebar.toggle(
        "Show prompts", False, on_change=set_show_prompts)

    st.session_state.current_question = 1
    st.session_state.show_prompt = False
    st.session_state.initial_prompt = (
        f'Make a {drawing_style} drawing.  {question_data["initial_image_prompt"]}'
    )
    st.session_state.image_history = [st.session_state.initial_prompt]
    st.session_state.responses = []
    st.session_state.question_data = question_data
    st.session_state.yaml_path = yaml_path
    st.session_state.questions = question_data["questions"]
    with st.spinner(f"Generating initial image with {st.session_state.ai}..."):
        st.sidebar.write("I am creating an initial image for you to ponder...")
        generated_image = generate_image(st.session_state.image_history)
        if generated_image is not None:
            st.session_state.current_image = generated_image

    st.sidebar.write("Here is an image to contemplate to start your journey.")
    if st.session_state.show_prompts:
        st.sidebar.write(st.session_state.initial_prompt)
    st.sidebar.write(
        "Let's begin!  After every question, select the emotion and image that resonates with you the most."
    )

# Display current question

if st.session_state.current_question < len(st.session_state.questions):
    st.image(st.session_state.current_image, caption="Your current image")
    question = st.session_state.questions[st.session_state.current_question]
    st.write(
        f"## {st.session_state.current_question}: **{question['question']}**")

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
                + f"The viewer should feel {response['emotion'].lower()}."
            )
            with st.spinner(
                f"Creating image for {response['emotion'].lower()} using {st.session_state.ai}..."
            ):
                st.sidebar.write(
                    f"Creating image for {response['emotion']} using {st.session_state.ai}..."
                )
                generated_image = generate_image([prompt])
            if generated_image is not None:
                st.button(
                    emo,
                    on_click=select_image,
                    args=(emo, prompt, generated_image, response),
                    type="primary"
                )
                st.image(generated_image, caption=response["emotion"])
                if st.session_state.show_prompt:
                    st.write(prompt)

            else:
                st.error("An error occurred while generating the image.")

else:
    # Show final image and summary
    st.sidebar.write(
        "Thank you for completing the journey together. Here is the final image representing your journey:"
    )

    if st.session_state.current_image is not None:
        st.image(
            st.session_state.current_image, caption="Your Climate Anxiety Reflection"
        )

    # Show summary of emotions
    st.sidebar.write("### Your Emotional Journey")
    for i, (question, response) in enumerate(
        zip(st.session_state.questions, st.session_state.responses)
    ):
        st.sidebar.write(
            f"**Question {i+1}**: {question['question']} => {response['emotion']}"
        )

    # Save image option
    st.button("Save Image", on_click=save_image,
              args=(st.session_state.current_image,))

    st.button("Start Over", on_click=start_over)

    # Option to email the image (for future implementation)
    # email = st.text_input("Enter your email to receive the final image and summary:")
    # if st.button("Send Email"):
    #     send_email(email, final_image, st.session_state.responses)

# Note: Replace 'questions.yaml' with the actual path to your YAML file containing the questions and prompts.
