import torch,os
from pathlib import Path


task = 'topic'
dataset = 'agnews'

SEP_TOKEN = '[SEP]'
CLS_TOKEN = '[CLS]'
TRAIN_FILE_PATH = '../../data/'+task+'/'+dataset+'/IMDB Dataset.csv'
#VAL_FILE_PATH = '../../data/'+task+'/'+dataset+'/test.csv'
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 4
NUM_EPOCHS = 6
GRADIENT_ACCUMULATION_STEPS = 8
WARMUP_STEPS = 3
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

FINETUNED_MODEL_SAVE_PATH="../../models/CS/"+task+"/"+dataset+"/"
PRETRAINED_MODEL_PATH= "../../models/bert-base-cased"
my_file = Path(FINETUNED_MODEL_SAVE_PATH)

if not my_file.exists():
    os.makedirs(FINETUNED_MODEL_SAVE_PATH)