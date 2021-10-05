import os, sys, argparse, json, time, numpy
from tqdm import tqdm
from easydict import EasyDict as edict
import torch, torchvision
import torch.nn as nn
import torch.optim as optim
from COCO_Captions_process import image_text_dataset
from CrossModalTransformer import CrossModalTransformer
from transformers import BertTokenizer, BertTokenizerFast

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

def get_pretraining_args():
    desc = "shared config for pretraining and finetuning"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("image_dir", type=str, help="eg: /data/wangyan/COCO_Captions/train2014/")
    parser.add_argument("annotation_file", type=str, help="eg: ./COCO_Captions_process/coco_annotations.txt")
    parser.add_argument("--pretrained_visual_weights", type=str, help="pretrained visual weights",
                        default="./pretrained/grid_feat_R-50.pth")
    parser.add_argument("--pretrained_txt_weights", type=str, help="pretrained textual weights",
                        default="./pretrained/pretrained_txt_weights.pth")
    parser.add_argument("--tokenizer_dir", type=str, help="path to tokenizer dir",
                        default="./pretrained/bert-base-uncased/")
    parser.add_argument("--result_dir", type=str, default="./result",
        help="dir to store model checkpoints & training meta.")
    parser.add_argument("--pretrained_model_file_name", type=str, default="/image_txt_pretrained_model.ckpt",
        help="pretrained model file name, don't forget a '/' ahead.")
    parser.add_argument("--best_pretrained_model_file_name", type=str, default="/best_image_txt_pretrained_model.ckpt",
        help="best pretrained model file name, don't forget a '/' ahead.")
    parser.add_argument("--config", default="./hyper_param.json", help="JSON config files")
    parsed_args = parser.parse_args()
    args = edict(vars(parsed_args))
    config_args = json.load(open(args.config))
    override_keys = {arg[2:].split("=")[0] for arg in sys.argv[1:]
                     if arg.startswith("--")}
    for k, v in config_args.items():
        if k not in override_keys:
            setattr(args, k, v)
    return args

config = get_pretraining_args()

def my_collate_fn(batch):
    # input: a list of bsz dicts,
    # the structure of each one(batch) is:
    # 'image': a tensor and a string: [C, H, W]
    # 'text': ['A very clean and well decorated empty bathroom']
    # print(len(batch))

    crop = torchvision.transforms.CenterCrop([224, 224])

    batch_tensor = {}
    batch_tensor['image'] = torch.empty(config.bsz, 3, 224, 224)
    batch_tensor['text'] = []
    for i, image_text_pairs in enumerate(batch):
        batch_tensor['image'][i] = crop(image_text_pairs['image'])
        batch_tensor['text'].append(image_text_pairs['text'])

    # print(len(batch_tensor))
    return batch_tensor

def start_cross_modal_pretrain():
    scaler = torch.cuda.amp.GradScaler()
    autocast = torch.cuda.amp.autocast

    tokenizer = BertTokenizerFast.from_pretrained(config.tokenizer_dir)

    root = os.path.join(config.image_dir)
    annotation_file = os.path.join(config.annotation_file)

    dataset = image_text_dataset.ImageTextDataset(
        root_path=root,
        annotationfile_path=annotation_file,
        transform=image_text_dataset.ImglistToTensor(),
        random_shift=True,
        test_mode=False
    )

    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=config.bsz, shuffle=False,
                                               collate_fn=my_collate_fn, num_workers=4)
    model = CrossModalTransformer(config)
    #model.to(device)
    model = torch.nn.DataParallel(model, device_ids=config.DEVICE_IDS)  # 声明所有可用设备
    model = model.cuda(device=config.DEVICE_IDS[0])  # 模型放在主设备
    model.requires_grad_(True)

    visual_checkpoint = torch.load(config.pretrained_visual_weights)
    model.module.frame_encoder.load_state_dict(visual_checkpoint, False)

    txt_checkpoint = torch.load(config.pretrained_txt_weights)
    model.module.query_embed.load_state_dict(txt_checkpoint, False)

    loss_func = nn.L1Loss(reduction="mean")
    optimizer = optim.SGD(params=model.parameters(), lr=config.learning_rate)
    optimizer.zero_grad()

    result_dir = config.result_dir
    train_log_filename = "cross_modal_pretrain_log.txt"
    train_log_filepath = os.path.join(result_dir, train_log_filename)
    train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"

    min_loss = 1000

    for epoch in range(config.num_epochs):

        epoch_time_start = time.time()
        for i, batch in tqdm(enumerate(train_loader),
                          desc="Training batch iteration",
                          total=len(train_loader)):
            #batch structure:
            #'image': bsz, C, H, W
            #'text': list of bsz

            word = batch["text"]
            word_token = tokenizer(word, padding=True, return_tensors="pt")
            if(len(batch["text"])!=config.bsz):
                word_token['input_ids'] = torch.tensor(numpy.pad(word_token['input_ids'], ((0, config.bsz-len(batch["text"])), (0, 0))))

            image = batch["image"].cuda(device=config.DEVICE_IDS[0])

            with autocast():

                pos_scores = model(image, word_token['input_ids']).cuda(device=config.DEVICE_IDS[0])
                sum_pos_scores = torch.sum(torch.sum(pos_scores, dim=-1), dim=-1)

                idx = torch.randint(1, config.bsz, size=(config.bsz,)).cuda(device=config.DEVICE_IDS[0])
                neg_scores = model(image, word_token['input_ids'][idx]).cuda(device=config.DEVICE_IDS[0])
                sum_neg_scores = torch.sum(torch.sum(neg_scores, dim=-1), dim=-1)

                loss = (torch.clamp(config.margin + sum_neg_scores - sum_pos_scores, min=0).sum() / config.bsz)\
                    .cuda(device=config.DEVICE_IDS[0])

            print("\nloss:", float(loss))
            optimizer.zero_grad()
            #loss.backward()
            scaler.scale(loss).backward()
            #optimizer.step()
            scaler.step(optimizer)
            scaler.update()

            print("------this batch time: %.1fs------" % (time.time() - epoch_time_start))

        if loss < min_loss:
            min_loss = loss
            checkpoint = {
                "model_param": model.state_dict(),
                "model_cfg": config}
            torch.save(checkpoint, result_dir + config.best_pretrained_model_file_name)

        to_write = train_log_txt_formatter.format(time_str=time.strftime("%Y_%m_%d_%H:%M:%S"),
                                                  epoch=epoch,
                                                  loss_str=" ".join(["{}".format(loss)]))
        with open(train_log_filepath, "a") as f:
            f.write(to_write)
        print("......one epoch time: %.1fs......" % (time.time() - epoch_time_start))

    checkpoint = {
        "model_struct": model,
        "model_param": model.state_dict(),
        "model_cfg": config}
    torch.save(checkpoint, result_dir + config.pretrained_model_file_name)


if __name__ == '__main__':
    start_cross_modal_pretrain()