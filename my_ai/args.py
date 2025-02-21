import argparse
import os.path

from my_ai.core import AiCore
from my_ai.train import Training
from my_ai.chat import Chatbot

def run_gpt(hyperparameters):
    print("Booting up GPT")
    Chatbot(hyperparameters)

def train_gpt(hyperparameters):
    print("Starting train")
    Training(hyperparameters)

def automated_routine(hyperparameters):
    if os.path.isfile('vocab.txt'):
        "In order to run this program you must use one of the following parameters:\n -r : start the prompt, but it'll only work if you already have a Vocab and a Model file \n -t : start training the AI, don't forget to review the hyperparameters and the folder containing the files."
    else:
        print('There is no vocab file')

def parse_args():
    parser = argparse.ArgumentParser(description='LLM starter')
    parser.add_argument('-r', action='store_true', help='Run the GPT')
    parser.add_argument('-t', action='store_true', help='Train the GPT')
    return parser.parse_args()

def script(hyperparameters):
    args = parse_args()
    
    if args.r:
        run_gpt(hyperparameters)
    elif args.t:
        train_gpt(hyperparameters)
    else:
        automated_routine(hyperparameters)