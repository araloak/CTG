import torch


task = 'sentiment'
dataset = 'imdb'

SEP_TOKEN = '[SEP]'
CLS_TOKEN = '[CLS]'
TRAIN_FILE_PATH = '../../data/'+task+'/'+dataset+'/IMDB Dataset.csv'
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 4
NUM_EPOCHS = 6
GRADIENT_ACCUMULATION_STEPS = 8
WARMUP_STEPS = 3
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

FINETUNED_MODEL_SAVE_PATH="../../models/CS/"+task+"/"+dataset+"/"
PRETRAINED_MODEL_PATH= "../../models/bert-base-cased"
