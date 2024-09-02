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

# Directory where YAML files are stored
questions_directory = "questions"


# Function to load questions and prompts from a YAML file
def load_questions(yaml_file):
    with open(yaml_file, "r") as file:
        return yaml.safe_load(file)


# MAGIC_PROMPT = "I NEED to test how the tool works with extremely simple prompts. DO NOT add any detail, just use it AS-IS: "
MAGIC_PROMPT = ""


# Function to generate or modify an image with DALL-E
def generate_image(prompt_chain):
    full_prompt = MAGIC_PROMPT + " ".join(prompt_chain)
    new_prompt = ensure_prompt_length(full_prompt)
    if new_prompt != full_prompt:
        st.sidebar.write(
            f"The prompt was too long and has been shortened to the following: {new_prompt}"
        )
    response = client.images.generate(
        prompt=new_prompt, n=1, size="1024x1024", model="dall-e-3"
    )
    image_url = response.data[0].url
    return Image.open(BytesIO(requests.get(image_url).content))


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


def ensure_prompt_length(prompt, max_length=1024):
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


# Display Butterfly Assistant
st.sidebar.image("butterfly.png", use_column_width=True)
st.sidebar.write(
    "Hi! I'm your Butterfly Assistant. I'll guide you through this experience."
)

# Set up the UI for YAML file selection
st.title("ClimAnx")
st.subheader("Explore your climate anxiety through AI-generated images.")

# Initialize session state
if (
    "current_question" not in st.session_state
    or st.session_state.current_question == -1
):
    # Get the list of YAML files
    yaml_files = get_yaml_files(questions_directory)
    selected_yaml = st.sidebar.selectbox("Choose a questionnaire:", yaml_files, None)

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
            "Photo",
            "Cartoon",
            "Abstract",
            "Expressionist",
            "Cubist",
            "Impressionist",
            "Naturalist",
            "Ukiyo-e",
            "Medieval",
            "Futurist",
            "Fantasy",
            "Fauvist",
            "Comic Strip",
        ],
        None,
    )
    if not drawing_style:
        st.sidebar.write("Please select a drawing style.")
        st.stop()

    st.session_state.drawing_style = drawing_style.lower()
    st.session_state.current_question = 0
    st.session_state.image_history = [
        f'Make a {drawing_style} drawing.  {question_data["initial_image_prompt"]}',
    ]
    st.session_state.responses = []
    st.session_state.question_data = question_data
    st.session_state.yaml_path = yaml_path
    st.session_state.questions = question_data["questions"]
    with st.spinner("Generating initial image..."):
        st.sidebar.write("I am creating an initial image for you to ponder...")
        generated_image = generate_image(
            f"Make a {drawing_style} drawing.  {st.session_state.image_history}"
        )
        st.session_state.current_image = generated_image

    st.sidebar.write("Here is an image to contemplate to start your journey.")
    st.sidebar.write(
        "Let's begin!  After every question, select the emotion that resonates with you the most."
    )

# Display current question

if st.session_state.current_question < len(st.session_state.questions):
    st.image(st.session_state.current_image, caption="Your current image")
    question = st.session_state.questions[st.session_state.current_question]
    st.write(f"{st.session_state.current_question}: **{question['question']}**")

    # Show options as small images
    ncols = len(question["responses"])
    base_prompt = " ".join(st.session_state.image_history)
    cols = st.columns(ncols)
    for i, response in enumerate(question["responses"]):
        with cols[i]:
            prompt = (
                base_prompt
                + " "
                + response["prompt"]
                + " "
                + f"I feel {response['emotion']}."
            )
            with st.spinner(f"Creating image for {response['emotion']}..."):
                st.sidebar.write(f"Creating image for {response['emotion']}...")
                generated_image = generate_image(prompt)
            st.image(generated_image, caption=response["emotion"])
            st.write(prompt)
            emo = response["emotion"]
            st.button(
                emo,
                on_click=select_image,
                args=(emo, prompt, generated_image, response),
            )

else:
    # Show final image and summary
    st.sidebar.write(
        "Thank you for completing the journey together. Here is the final image representing your journey:"
    )

    st.image(st.session_state.current_image, caption="Your Climate Anxiety Reflection")

    # Show summary of emotions
    st.sidebar.write("### Your Emotional Journey")
    for i, response in enumerate(st.session_state.responses):
        st.sidebar.write(f"**Question {i+1}**: {response['emotion']}")

    # Save image option
    st.button("Save Image", on_click=save_image, args=(st.session_state.current_image,))

    st.button("Start Over", on_click=start_over)

    # Option to email the image (for future implementation)
    # email = st.text_input("Enter your email to receive the final image and summary:")
    # if st.button("Send Email"):
    #     send_email(email, final_image, st.session_state.responses)

# Note: Replace 'questions.yaml' with the actual path to your YAML file containing the questions and prompts.
