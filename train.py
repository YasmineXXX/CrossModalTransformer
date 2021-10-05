import os
import torch, torchvision
import torch.nn as nn
import torch.optim as optim
import argparse, json, sys, time, numpy
from tqdm import tqdm
from easydict import EasyDict as edict
from torch.utils.data.dataloader import default_collate
from transformers import BertTokenizer, BertTokenizerFast
from Charades_STA_process import charades_dataset
from CrossModalTransformer import CrossModalTransformer


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
#DEVICE = "cuda:0"
#device = torch.device(DEVICE)

def get_pretraining_args():
    desc = "shared config for pretraining and finetuning"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("frame_dir", type=str, help="eg: /data/wangyan/Charades-STA/Charades_v1_rgb/")
    parser.add_argument("annotation_file", type=str, help="eg: ./Charades_STA_process/charades_annotations.txt")
    parser.add_argument("--pretrained_visual_weights", type=str, help="pretrained visual weights",
                        default="./pretrained/grid_feat_R-50.pth")
    parser.add_argument("--pretrained_txt_weights", type=str, help="pretrained textual weights",
                        default="./pretrained/pretrained_txt_weights.pth")
    parser.add_argument("--pretrained_image_txt_weights", type=str, help="pretrained textual weights",
                        default="./pretrained/image_txt_pretrained_model.ckpt")
    parser.add_argument("--tokenizer_dir", type=str, help="path to tokenizer dir",
                        default="./pretrained/bert-base-uncased/")
    parser.add_argument("--result_dir", type=str, default="./result",
        help="dir to store model checkpoints & training meta.")
    parser.add_argument("--best_train_model_file_name", type=str, default="/best_model.ckpt",
        help="best train model file name, don't forget a '/' ahead.")
    parser.add_argument("--train_model_file_name", type=str, default="/model.ckpt",
        help="train model file name, don't forget a '/' ahead.")
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
    # 'frame_label': a tensor and a string:
    #        [fms, C, H, W]   ('AO8RW',)
    # 'query_sent': ['a person is putting a book on a shelf.']
    # 'start_frame': tensor([1])
    # 'end_frame': tensor([809])
    # 'clip_start_frame': tensor([0])
    # 'clip_end_frame': tensor([165])
    # print(len(batch))

    fms_list = [e['frames_label'][0].shape[0] for e in batch]
    max_fms = max(fms_list)
    pack_fms = [(max_fms - f) for f in fms_list]

    crop = torchvision.transforms.CenterCrop([224, 224])

    batch_tensor = {}
    batch_tensor['frames'] = torch.empty(config.bsz, max_fms, 3, 224, 224)
    batch_tensor['video_name'] = []
    batch_tensor['query_sent'] = []
    batch_tensor['start_frame'] = torch.empty(config.bsz)
    batch_tensor['end_frame'] = torch.empty(config.bsz)
    batch_tensor['clip_start_frame'] = torch.empty(config.bsz)
    batch_tensor['clip_end_frame'] = torch.empty(config.bsz)
    for i, video in enumerate(batch):
        batch_tensor['frames'][i] = torch.empty(max_fms, 3, 224, 224)
        for frame in range(video['frames_label'][0].shape[0]):
            batch_tensor['frames'][i][frame] = crop(video['frames_label'][0][frame])
        torch.nn.functional.pad(batch_tensor['frames'][i], (0, 0, 0, 0, 0, 0, 0, fms_list[i]))
        batch_tensor['video_name'].append(video['frames_label'][1])
        batch_tensor['query_sent'].append(video['query_sent'])
        batch_tensor['start_frame'][i] = video['start_frame']
        batch_tensor['end_frame'][i] = video['end_frame']
        batch_tensor['clip_start_frame'][i] = video['clip_start_frame']
        batch_tensor['clip_end_frame'][i] = video['clip_end_frame']

    # print(len(batch_tensor))
    return batch_tensor

def maxlist(nums):
    dp = nums.copy()
    index_dp = [0] * len(nums)
    dp[0] = nums[0]
    seq = []
    for i in range(1, len(nums)):
        A_now = float(nums[i])
        dp[i] = max(A_now, A_now + dp[i - 1])

        if dp[i - 1] < 0:
            index_dp[i] = i
        else:
            index_dp[i] = index_dp[i - 1]

    sum_max = max(dp)  # 最大和
    seq_max = dp.index(sum_max)  # 最大和所在的位置

    seq_sum_max = nums[index_dp[seq_max]:seq_max + 1]  # 最大和的子序列

    seq_sum_index = []
    for i in range(len(seq_sum_max)):
        seq_sum_index.append(nums.index(seq_sum_max[i]))
    seq_sum_index

    return seq_sum_index


def start_training():
    scaler = torch.cuda.amp.GradScaler()
    autocast = torch.cuda.amp.autocast

    # tokenizer = BertTokenizer.from_pretrained(config.tokenizer_dir)
    tokenizer = BertTokenizerFast.from_pretrained(config.tokenizer_dir)

    result_dir = config.result_dir
    '''
    #small test dataset loading
    root = os.path.join(os.getcwd(),
                        'VideoDatasetLoadingPytorch/my_dataset')  # Folder in which all videos lie in a specific structure
    annotation_file = os.path.join(root,
                                   'annotations.txt')  # A row for each video sample as: (VIDEO_PATH START_FRAME END_FRAME CLASS_ID)

    dataset = my_video_dataset.VideoFrameDataset(
        root_path=root,
        annotationfile_path=annotation_file,
        segment_duration=config.segment_duration,
        imagefile_template='img_{:05d}.jpg',
        transform=my_video_dataset.ImglistToTensor(),
        random_shift=True,
        test_mode=False
    )
    '''
    # Charades-STA dataset loading
    root = os.path.join(config.frame_dir)
    annotation_file = os.path.join(config.annotation_file)

    dataset = charades_dataset.VideoFrameDataset(
        root_path=root,
        annotationfile_path=annotation_file,
        segment_duration=config.segment_duration,
        imagefile_template='img_{:06d}.jpg',
        transform=charades_dataset.ImglistToTensor(),
        random_shift=True,
        test_mode=False
    )

    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=config.bsz, shuffle=False,
                                               collate_fn=my_collate_fn)
    model = CrossModalTransformer(config)
    # model.to(device)
    model = torch.nn.DataParallel(model, device_ids=config.DEVICE_IDS)  # 声明所有可用设备
    model = model.cuda(device=config.DEVICE_IDS[0])  # 模型放在主设备
    model.requires_grad_(True)

    visual_checkpoint = torch.load(config.pretrained_visual_weights)
    model.module.frame_encoder.load_state_dict(visual_checkpoint, False)

    txt_checkpoint = torch.load(config.pretrained_txt_weights)
    model.module.query_embed.load_state_dict(txt_checkpoint, False)

    image_txt_checkpoint = torch.load(config.pretrained_image_txt_weights)
    model.module.multihead_att.load_state_dict(image_txt_checkpoint["model_param"], False)

    loss_func = nn.L1Loss(reduction="mean")
    optimizer = optim.SGD(params=model.parameters(), lr=config.learning_rate)
    optimizer.zero_grad()

    train_log_filename = "train_log.txt"
    train_log_filepath = os.path.join(result_dir, train_log_filename)
    train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"

    min_loss = 10000

    for epoch in range(config.num_epochs):

        epoch_time_start = time.time()
        for i, batch in tqdm(enumerate(train_loader),
                             desc="Training batch iteration",
                             total=len(train_loader)):
            # batch structure:
            # 'frames': bsz, fms, C, H, W
            # 'video_name': list of bsz
            # 'query_sent': list of bsz
            # 'start_frame': bsz
            # 'end_frame': bsz
            # 'clip_start_frame': bsz
            # 'clip_end_frame': bsz
            word = batch["query_sent"]
            word_token = tokenizer(word, padding=True, return_tensors="pt")
            # word_token = tokenizer(word, padding=True, return_tensors="pt").cuda(device=DEVICE_IDS[0])
            # word_pos_embed = model_text_proc(word_token['input_ids'])

            # frames = batch["frames"].to(device)  # bsz, fms, C, H, W
            frames = batch["frames"].cuda(device=config.DEVICE_IDS[0])

            fms = frames.size(1)
            C = frames.size(2)
            H = frames.size(3)
            W = frames.size(4)
            frames = frames.view(-1, C, H, W)

            # with autocast():
            score = model(frames, word_token['input_ids'].repeat(fms, 1)).cuda(device=config.DEVICE_IDS[0])
            batch_frame_score = torch.sum(torch.sum(score, dim=-1), dim=-1)
            pred_start_batch = torch.empty(config.bsz).cuda(device=config.DEVICE_IDS[0])
            pred_end_batch = torch.empty(config.bsz).cuda(device=config.DEVICE_IDS[0])
            for idx in range(0, len(batch_frame_score), fms):
                seq_max_index = maxlist(list(batch_frame_score[idx:idx + fms]))
                pred_start_frame = seq_max_index[0] * config.segment_duration
                pred_end_frame = seq_max_index[-1] * config.segment_duration
                pred_start_batch[idx // fms] = pred_start_frame
                pred_end_batch[idx // fms] = pred_end_frame

            pred_start_batch.requires_grad_()
            pred_end_batch.requires_grad_()
            loss_start = loss_func(pred_start_batch, batch["clip_start_frame"].cuda(device=config.DEVICE_IDS[0]))
            loss_end = loss_func(pred_end_batch, batch["clip_end_frame"].cuda(device=config.DEVICE_IDS[0]))
            loss = (loss_start + loss_end) / config.bsz

            print("\nloss:", float(loss))
            optimizer.zero_grad()
            loss.backward()
            # scaler.scale(loss).backward()
            optimizer.step()
            # scaler.step(optimizer)
            # scaler.update()

            print("------this batch time: %.1fs------" % (time.time() - epoch_time_start))

        if loss < min_loss:
            min_loss = loss
            checkpoint = {
                "model_param": model.state_dict(),
                "model_cfg": config}
            torch.save(checkpoint, result_dir + config.best_train_model_file_name)

        '''torch.save({'epoch': epoch+1, 'state_dict': model.state_dict(), 'loss': loss,
                    'optimizer': optimizer.state_dict(), 'config:': config},
                   result_dir + str("%3d" % epoch) + 'pth.tar')'''
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
    torch.save(checkpoint, result_dir + config.train_model_file_name)


if __name__ == '__main__':
    start_training()

