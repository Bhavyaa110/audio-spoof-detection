import torch
SAMPLE_RATE = 16000
DURATION = 3   # change to 5 for 5 sec
AUDIO_LEN = SAMPLE_RATE * DURATION

BATCH_SIZE = 16
EPOCHS = 30
LR = 0.0001

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "outputs/deeprawnet.pth"