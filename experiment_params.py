# --- DEMOGRAPHIC AND EXPERIMENT PARAMETERS ---
from pathlib import Path
# Demographic questions configuration
DEMOGRAPHIC_VARIABLES = [
    {'name': 'age', 'prompt': 'Please enter your age:', 'type': 'number'},
    {'name': 'gender', 'prompt': 'Please select your gender:', 'type': 'choice', 'choices': ['Male', 'Female', 'Non-binary']},
    {'name': 'education', 'prompt': 'What is your highest educational achievement?', 'type': 'choice', 'choices': ['High school diploma', \
        "Bachelor's degree", "Master's degree", 'PhD/Doctoral degree']}
]
# Experiment parameters (set here instead of using a dialog)
PARTICIPANT_ID = "test_participant"  # Change as needed
SINGLE_IMAGE_DURATION = 10  # in seconds, e.g., 5, 6, or 10
BREATH_EXERCISE_DURATION = 10  # in seconds, e.g., 10, 120, or 180
BREATH_IN_DURATION = 4  # in seconds, e.g., 4
BREATH_OUT_DURATION = 6  # in seconds, e.g., 6
CONSECUTIVE_BLOCKS = 1  # e.g., 1, 2, 3, or 5
NBR_OF_BLOCKS = 3
RESOLUTION = [1920, 1200]
# Target emotions for experiment
TARGET_EMOTIONS = ["Happiness", "Sadness", "Anger"]
# List of emotions used in the experiment
EMOTIONS_LIST = [
    "Happiness",
    "Sadness",
    "Anger",
    "Contempt",
    "Confusion",
    "Fear",
    "Surprise",
    "Calmness",
    "Neutral"
]

BASE_IMAGE_DIR = Path(r"C:\Users\--\Documents\Alexithymia\image")
BASE_AUDIO_DIR = Path(r"C:\Users\--\Documents\Alexithymia\audio")
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
AUDIO_EXTS = {'.mp3', '.wav', '.ogg', '.flac', '.m4a', '.aac'}
SINGLE_AUDIO_DURATION = 60  # in seconds, default for audio trials
N_IMAGES_PER_BLOCK = 6
N_AUDIO_PER_BLOCK = 3

# Instruction text for the emotion question
EMOTION_INSTRUCTION = "Use the mouse or number keys to select one emotion, then press Enter."
# Instruction text for the confidence rating question
CONFIDENCE_INSTRUCTION = "Use the mouse or arrow keys to select a value, then press Enter."

# PANAS Questionnaire items
PANAS_ITEMS = [
    "Interested",
    "Distressed",
    "Excited",
    "Upset",
    "Strong",
    "Guilty",
    "Scared",
    "Hostile",
    "Enthusiastic",
    "Proud",
    "Irritable",
    "Alert",
    "Ashamed",
    "Inspired",
    "Nervous",
    "Determined",
    "Attentive",
    "Jittery",
    "Active",
    "Afraid"
]
# TAS and AQ questionnaire questions
TAS20_QUESTIONS = [
    "I am often confused about what emotion I am feeling.",
    "It is difficult for me to find the right words for my feelings.",
    "I have physical sensations that even doctors don’t understand.",
    "I am able to describe my feelings easily.",
    "I prefer to analyze problems rather than just describe them.",
    "When I am upset, I don’t know if I am sad, frightened, or angry.",
    "I am often puzzled by sensations in my body.",
    "I prefer to just let things happen rather than to understand why they turned out that way.",
    "I have feelings that I can’t quite identify.",
    "Being in touch with emotions is essential.",
    "People tell me to describe my feelings more.",
    "I don’t know what’s going on inside me.",
    "I often don’t know why I am angry.",
    "I prefer talking to people about their daily activities rather than their feelings.",
    "I prefer to watch “light” entertainment shows rather than psychological dramas.",
    "It is difficult for me to reveal my innermost feelings, even to close friends.",
    "I can feel close to someone, even in moments of silence.",
    "I find examination of my feelings useful in solving personal problems.",
    "Looking for hidden meanings in movies or plays distracts from my enjoyment."
]
AQ_10_QUESTIONS = [
    "I often notice small sounds when others do not.",
    "When I’m reading a story, I find it difficult to work out the characters’ intentions.",
    "I find it easy to 'read between the lines' when someone is talking to me.",
    "I usually concentrate more on the whole picture, rather than the small details.",
    "I know how to tell if someone listening to me is getting bored.",
    "I find it easy to do more than one thing at once.",
    "I find it easy to work out what someone is thinking or feeling just by looking at their face.",
    "If there is an interruption, I can switch back to what I was doing very quickly.",
    "I like to collect information about categories of things.",
    "I find it difficult to work out people’s intentions."
]

# Likert scale for AQ-10 and TAS-20 answers
LIKERT_SCALE = [
    "1 = definitely agree",
    "2 = slightly agree",
    "3 = I don't know",
    "4 = slightly disagree",
    "5 = definitely disagree"
]

# PANAS rating options
PANAS_RATING = [
    "1 = very slightly or not at all",
    "2 = a little",
    "3 = moderately",
    "4 = quite a bit",
    "5 = extremely"
]
# Instruction text for Likert scale questionnaires
QUESTIONNAIRE_INSTRUCTION = "Use the mouse or number keys to select your answer, then press Enter to continue."

# Welcome and demographic instruction text
WELCOME_TEXT = "Welcome to our experiment. Thank you for participating!"
DEMOGRAPHIC_INSTRUCTION_TEXT = (
    "Before we begin, we would like to collect some demographic information.\n"
    "On the next screen, please enter your age, gender, and highest educational achievement. Then press Enter to submit your answers.\n"
    "Please press Enter to continue."
)
# Start experiment prompt text
START_EXPERIMENT_TEXT = "The experiment can start now. Please press 'ENTER' whenever you're ready to start."

# Transition screen text before questionnaires
TRANSITION_TEXT = (
    "You have reached the second part of the experiment.\n\n"
    "Please answer the following 3 questionnaires.\n\n"
    "Press Enter to continue."
)

# Final thank you text for the end of the experiment
THANK_YOU_TEXT = (
    "Thank you so much for your participation!\n\n"
    "Your time and effort are truly valuable to our research.\n\n"
    "The experiment is now complete.\n\n"
    "This window will close automatically in 10 seconds."
)
