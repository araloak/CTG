import torch
from pytorch_transformers import BertTokenizer, BertConfig
import  argparse

from BertModules import BertClassifier
from Constants import *
from Utils import seed_everything

# https://github.com/LawsonAbs/learn/blob/master/bert/%E4%BD%BF%E7%94%A8bert%E5%BE%97%E5%88%B0attention.ipynb
parser = argparse.ArgumentParser(description='extract CNN pooling features from images')
parser.add_argument('--input_dir', default='../../data/'+task+'/'+dataset+'/test_data.txt')
parser.add_argument('--output_dir', default='../../data/'+task+'/'+dataset+'/test_data_result.txt')
args = parser.parse_args()

if task == "sentiment" and dataset == "imdb":
    NUM_LABEL = 2
elif task == "topic" and dataset == "agnews":
    NUM_LABEL = 4

def getAttention(output,layer,head):
    res = output[2]
    # print(type(res))# <class tuple>
    # print(len(res)) # 12 。解释一下是12的原因：Number of hidden layers in the Transformer encoder 是12
    # print(res) #(tensor()...,tensor()...)

    layer_attention_score = res[layer] # 得到 attention 中的最后一个tuple，也就是最后一层的attention 值
    #print(type(res)) # 去掉一维之后仍然是tuple <class 'tuple'>
    #print(attention_score.size()) # torch.Size([1, 12, 28, 28])。 这个size = [batch_size,num_head,seq_len,seq_len]

    layer_attention_score.squeeze_(0) # 去掉第一维的1
    #print(attention_score.size()) # 因为有12个head，这里只取第一个
    layer_head_attention_score = layer_attention_score[head,:,:]
    #print(layer_head_attention_score.size())
    #print(layer_head_attention_score)
    return layer_head_attention_score

def getTopWord(first_attention_score,index_of_sample=0):
    #making => 对应的下标是18。所以拿到18这行的行向量
    score = first_attention_score[18].tolist()
    #print(score)

    # 将上面的list转为dict，方便后面根据dict排序，然后找出相似度最高的index
    dic = {}
    for i in range(len(score)):
        dic[i] = score[i]
    #print(dic)

    # 将得到的dic 排序，值高的放在最前面
    res = sorted(dic.items(),key = lambda dic:dic[1],reverse=True)
    res1 = dict(res) # 排序后是list，再转为dict
    keys = res1.keys()
    #print(keys)

    # 得到tokens的id，并转为list
    tokens = input_ids[index_of_sample].squeeze_(0).tolist()
    i= 0
    for _ in keys:# 将token转换成相应的word
        if i < 5:
            print(bert_tokenizer.convert_ids_to_tokens( tokens[_]))
            i+=1
        else:
            break

def drawAttention(inputs,first_attention_score,head="mean"):
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib.ticker as ticker

    #将first_attention_score 的grad属性去除 => 使用detach()，并转为numpy
    res2 = first_attention_score.cpu().detach().numpy()
    temp = inputs.tolist() # 得到input_ids，为了将其转换成tokens
    #print(temp)
    tokens= bert_tokenizer.convert_ids_to_tokens(temp)
    #print(type(tokens)) # <class 'list'>
    #print(tokens) # ['[CLS]', 'it', 'is', ...]

    for i in range(len(tokens) - 1, -1,
                   -1):  # 倒序循环，从最后一个元素循环到第一个元素。不能用正序循环，因为正序循环删除元素后，后续的列表的长度和元素下标同时也跟着变了，由于len(alist)是动态的。
        if tokens[i] == '[PAD]':
            tokens.pop(i)
    tokens.remove('[SEP]')
    res2 = res2[:len(tokens),:len(tokens)]

    # tokens就是我们的横纵坐标(标签)
    df = pd.DataFrame(res2, columns=tokens, index=tokens)
    fig = plt.figure(figsize = (10,10))# 画图，这里调整了图片的大小，否则会因为word太长导致文字重叠
    ax = fig.add_subplot(1,1,1)
    cax = ax.matshow(df, interpolation='nearest', cmap='hot_r')
    fig.colorbar(cax)

    tick_spacing = 1
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.set_xticklabels([''] + list(df.columns))
    ax.set_yticklabels([''] + list(df.index))
    name = '../../data/'+task+'/'+dataset+'/test_data_attentions_'+str(head)+".png"
    plt.savefig(name) # 存储图片
    plt.show()

seed_everything()
bert_tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_PATH, do_lower_case=False)
config = BertConfig(hidden_size=768,
                    num_hidden_layers=12,
                    num_attention_heads=12,
                    intermediate_size=3072,
                    num_labels=NUM_LABEL,
                    do_lower_case=False
                   )

# Create our custom BERTClassifier model object
model = BertClassifier(config,PRETRAINED_MODEL_PATH).to(DEVICE)
state_dict = torch.load(FINETUNED_MODEL_SAVE_PATH+"/best_model.pt")
model.load_state_dict(state_dict)

model.eval()
lines = open(args.input_dir, encoding="utf8").readlines()

input_ids = []
segment_ids = []
input_mask = []
for words in lines:
    tokens = bert_tokenizer.tokenize(words)[:MAX_SEQ_LENGTH-3]
    tokens = [CLS_TOKEN] + tokens + [SEP_TOKEN]
    temp_input_ids = bert_tokenizer.convert_tokens_to_ids(tokens)

    # Segment ID for a single sequence in case of classification is 0.
    temp_segment_ids = [0] * len(temp_input_ids)

    # Input mask where each valid token has mask = 1 and padding has mask = 0
    temp_input_mask = [1] * len(temp_input_ids)

    # padding_length is calculated to reach max_seq_length
    padding_length = MAX_SEQ_LENGTH - len(temp_input_ids)
    temp_input_ids = temp_input_ids + [0] * padding_length
    temp_input_mask = temp_input_mask + [0] * padding_length
    temp_segment_ids = temp_segment_ids + [0] * padding_length

    input_ids += torch.tensor([temp_input_ids], dtype=torch.long, device=DEVICE)
    segment_ids += torch.tensor([temp_segment_ids], dtype=torch.long, device=DEVICE)
    input_mask += torch.tensor([temp_input_mask], device=DEVICE, dtype=torch.long)
    print("input_ids", len(input_ids))

input_ids = torch.stack(input_ids)
segment_ids = torch.stack(segment_ids)
input_masks = torch.stack(input_mask)

results = []
for input_id, segment_id, input_mask in zip(input_ids, segment_ids, input_masks):
    with torch.no_grad():
        all_head_attention_score = torch.zeros([512,512]).cuda()
        output, polarity = model.prob(input_ids=input_id.unsqueeze(0), token_type_ids=segment_id.unsqueeze(0),
                      attention_mask=input_mask.unsqueeze(0))
        _, predicted = torch.max(polarity, 1)
        print("plority score:",polarity)
        print("sentiment label:",predicted)

        layer_head_attention_score = getAttention(output, 11, 5)
        drawAttention(input_id, layer_head_attention_score, 5)

        for head in range(12):
            layer_head_attention_score = getAttention(output, 11, head)
            all_head_attention_score+=layer_head_attention_score
            drawAttention(input_id, layer_head_attention_score,head)
        # all_head_attention_score/=12
        # drawAttention(input_id, all_head_attention_score)
        break

# 写入结果
# with open(args.output_dir,"w",encoding="utf8") as f:
#     for score in results:
#         f.write(str(score.argmax(-1))+"\n")
#
# tso_label_score_list = [each.numpy() for each in tso_label_score_list]
# tso_label_score_list = np.stack(tso_label_score_list)
# np.save(tso_label_score_list,args.output+"/"+args.file_name,allow_pickle=True)
