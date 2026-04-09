import torch
SAMPLE_RATE = 16000
DURATION = 3   # change to 5 for 5 sec
AUDIO_LEN = SAMPLE_RATE * DURATION

BATCH_SIZE = 8
EPOCHS = 15
LR = 0.0001

DEVICE = torch.device("cpu") 
MODEL_PATH = "outputs/deeprawnet.pth"