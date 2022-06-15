import numpy as np
from transformers import AdamW
from data.dataset import corpus_dataset, read_questions_from_dir
from transformers import BertForSequenceClassification
import argparse
from torch.utils.data import DataLoader
import torch
import logging
import random
import torch.nn.functional as f
from transformers import AutoTokenizer

device = 'cpu'


def train(model, args, train_dataset, val_dataset, optimizer):
    train_dataloader = DataLoader(train_dataset, batch_size=8)
    val_dataloader = DataLoader(val_dataset, batch_size=1)
    for epoch in range(args.epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            
            optimizer.zero_grad()
            output = model(batch[0].to(device), token_type_ids=batch[1], attention_mask=(batch[2]).to(device),labels=(batch[3]).to(device))
            
            loss = f.cross_entropy(output[1],batch[3])
            logits = output[1]
            # 反向梯度信息
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪

            # 参数更新
            optimizer.step()
            label = torch.tensor(batch[3])
            acc = (logits.argmax(1) == label).float().mean()
            logging.info(f"Epoch: {epoch}, Batch[{step}/{len(train_dataloader)}], "
                        f"Train loss :{loss.item():.3f}, Train acc: {acc:.3f}")
        model.eval()
        with torch.no_grad():
            n = 0
            acc_sum = 0
            for _,test_data in enumerate(val_dataloader):
                test_output = model(test_data[0].to(device), token_type_ids=test_data[1], attention_mask=(test_data[2]).to(device),labels=test_data[3])
                _,pred = test_output[0], test_output[1]
                
                acc_sum += (pred.argmax(1) == test_data[3]).float().sum().item()
                n+=1

            torch.save(model.state_dict(),args.model_save_path)

            logging.info(f"Val Acc: {acc_sum/n}")

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def main():
    parser = argparse.ArgumentParser()

    #What's input like: is .txt file left: sentences right: class label
    parser.add_argument(
        "--data_dir",
        default='D:/project/NLP/ChatbotBert/ChatbotBert/chatterbot_corpus',
        type=str,
        help="The input data dir.",
    )
    parser.add_argument(
        "--val_dir",
        default='D:/project/NLP/ChatbotBert/ChatbotBert/val_data',
        type=str,
    )
    parser.add_argument(
        "--pretrained_model",
        default='bert-base-uncased',
        type=str,
        help="pretrained model path",
    )    
    parser.add_argument(
        "--epochs",
        default=20,
        type=int,
        help="pretrained model path",
    )   

    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--output_dir",
        default='outputs',
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--classes",
        default=9,
        type=int,
        help="The number of labels",
    )

    parser.add_argument(
        "--model_save_path",
        type=str,
        default='model.pt',
        help="model save path",
    )
    args = parser.parse_args()
    set_seed(args)
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("__main__")

    # dataset preparation
    model = BertForSequenceClassification.from_pretrained(args.pretrained_model, num_labels=args.classes, output_hidden_states=False)
    model.to(device)
    logger.info("Training/evaluation parameters %s", args)

    # training
    train_dataset = corpus_dataset(args.data_dir)
    val_dataset = corpus_dataset(args.val_dir)
    encoded_questions, label, label_dict, answer = read_questions_from_dir(args.data_dir)
    optimizer_params = {'lr': 1e-5, 'eps': 1e-6, 'correct_bias': False}

    optimizer = AdamW(model.parameters(), **optimizer_params)
    train(model, args, train_dataset, val_dataset, optimizer=optimizer)
    
    sentence = ['你平时喜欢做什么']

    print('问题：', sentence)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    sentence = tokenizer(sentence)  
    model.load_state_dict(torch.load('model.pt'))
    pred = model(torch.tensor(sentence['input_ids']).to(device),torch.tensor(sentence['token_type_ids']).to(device),torch.tensor(sentence['attention_mask']).to(device)) 
    pred = pred.logits.argmax(1)[0].item()
    
    print('回答：', random.choice(answer[pred]))




if __name__ == "__main__":
    main()
