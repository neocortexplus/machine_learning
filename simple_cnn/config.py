import argparse
from dotenv import load_dotenv
import os

def get_config():
    load_dotenv()
    parser = argparse.ArgumentParser(description='CNN Training Configuration')
    parser.add_argument('--batch_size', type=int, default=os.getenv('BATCH_SIZE', 32))
    parser.add_argument('--learning_rate', type=float, default=os.getenv('LEARNING_RATE', 0.01))
    parser.add_argument('--epochs', type=int, default=os.getenv('EPOCHS', 5))

    args = parser.parse_args()
    return args
