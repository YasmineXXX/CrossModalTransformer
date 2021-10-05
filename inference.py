import os, torch, argparse, time
import numpy as np
from tqdm import tqdm
from Charades_STA_process import charades_dataset
from train import my_collate_fn, maxlist, CrossModalTransformer
from transformers import BertTokenizer

def get_pretraining_args():
    desc = "inference config"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("frame_dir", type=str, help="eg: /data/wangyan/Charades-STA/Charades_v1_rgb/")
    parser.add_argument("annotation_file", type=str, help="eg: ./Charades_STA_process/charades_annotations_test.txt")
    parser.add_argument("--result_dir", type=str, default="./result",
                        help="dir to store model checkpoints & training meta.")
    parser.add_argument("--train_model_file_name", type=str, default="/model.ckpt",
                        help="train model file name, don't forget a '/' ahead.")
    parsed_args = parser.parse_args()
    return parsed_args

test_config = get_pretraining_args()

def compute_acuracy(pred_start_frame, pred_end_frame,
                    gt_start_frame, gt_end_frame, iou_thds=(0.1, 0.3, 0.5, 0.7)):
    # pred_start_frame: bsz
    # gt_start_frame: bsz
    intersection = np.maximum(0, np.minimum(pred_end_frame, gt_end_frame) - np.maximum(pred_start_frame, gt_start_frame))
    union = np.maximum(pred_end_frame, gt_end_frame) - np.minimum(pred_start_frame, gt_start_frame)  # not the correct union though
    iou_scores = np.divide(intersection, union, out=np.zeros_like(intersection), where=union != 0)
    iou_corrects_batch = {}
    for iou_thd in iou_thds:
        iou_corrects_batch[str(iou_thd)] = 0
    for iou_thd in iou_thds:
        iou_corrects_batch[str(iou_thd)] += (sum(iou_scores >= iou_thd))
    return iou_corrects_batch

def start_inference():
    checkpoint = torch.load(test_config.result_dir + test_config.train_model_file_name)
    train_config = checkpoint["model_cfg"]

    root = os.path.join(test_config.frame_dir)
    annotation_file = os.path.join(test_config.annotation_file)

    test_dataset = charades_dataset.VideoFrameDataset(
        root_path=root,
        annotationfile_path=annotation_file,
        segment_duration=train_config.segment_duration,
        imagefile_template='img_{:06d}.jpg',
        transform=charades_dataset.ImglistToTensor(),
        random_shift=True,
        test_mode=False
    )

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=train_config.bsz, shuffle=False,
                                               collate_fn=my_collate_fn)

    model = CrossModalTransformer(train_config)
    model.load_state_dict(checkpoint["model_param"], False)
    model.eval()
    #model.frame_encoder.load_state_dict()
    model = torch.nn.DataParallel(model, device_ids=train_config.DEVICE_IDS)  # 声明所有可用设备
    model = model.cuda(device=train_config.DEVICE_IDS[0])  # 模型放在主设备

    tokenizer = BertTokenizer.from_pretrained(train_config.tokenizer_dir)
    inference_start_time = time.time()
    iou_thds = (0.1, 0.3, 0.5, 0.7)
    iou_corrects = {}
    for iou_thd in iou_thds:
        iou_corrects[str(iou_thd)] = 0
    test_log_filename = "test_log.txt"
    test_log_filepath = os.path.join(train_config.result_dir, test_log_filename)
    test_log_txt_formatter = "{time_str} [Accuracy] {acc:s}\n"
    for i, batch in tqdm(enumerate(test_loader),
                         desc="Inference batch iteration",
                         total=len(test_loader)):
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

        frames = batch["frames"].cuda(device=train_config.DEVICE_IDS[0])

        frame = torch.split(frames, 1, dim=1)  # bsz, 1, C, H, W

        # for j in range(len(frames)):
        # print(j, frames[j].squeeze(1).shape)
        video = torch.empty((train_config.bsz, len(frame)))

        for j in range(len(frame)):
            # model input: bsz, C, H, W
            # model output: bsz, C, H, W
            # result_ResNet = model(frame[j].squeeze(1))
            # print(result_ResNet)
            # result_pe = model_visual_embed(result_ResNet)
            frame[j].requires_grad_()
            # score: bsz, 1
            score = model(frame[j].squeeze(1), word_token['input_ids'])
            batch_frame_score = torch.sum(torch.sum(score, dim=-1), dim=-1)
            # print(score.shape)
            # print(att)
            # print(batch_frame_score.shape)
            video[:, j] = batch_frame_score

        pred_start_batch = torch.empty(train_config.bsz)

        pred_end_batch = torch.empty(train_config.bsz)


        for k in range(len(video)):
            seq_max_index = maxlist(list(video[k, :]))
            pred_start_frame = seq_max_index[0] * train_config.segment_duration
            pred_end_frame = seq_max_index[-1] * train_config.segment_duration
            pred_start_batch[k] = pred_start_frame
            pred_end_batch[k] = pred_end_frame

        iou_corrects_batch = compute_acuracy(pred_start_frame, pred_end_frame, batch["clip_start_frame"], batch["clip_end_frame"])
        for iou_thd in iou_thds:
            iou_corrects[str(iou_thd)] += iou_corrects_batch[str(iou_thd)]
        print("Now Accuracy:")
        for iou_thd in iou_thds:
            print("%.2f%%" % (iou_corrects[str(iou_thd)] / (train_config.bsz * (i+1)) * 100))

    acc = ""
    for iou_thd in iou_thds:
        iou_corrects[str(iou_thd)] /= len(test_loader)
        iou_corrects[str(iou_thd)] *= 100
        acc += "iou=%s: %s%%, " % (iou_thd, iou_corrects[str(iou_thd)])

    to_write = test_log_txt_formatter.format(time_str=time.strftime("%Y_%m_%d_%H:%M:%S"), acc=acc)
    with open(test_log_filepath, "a") as f:
        f.write(to_write)
    print("......inference time: %.1fs......" % (time.time() - inference_start_time))

if __name__ == '__main__':
    start_inference()
