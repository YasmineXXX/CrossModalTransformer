import json

if __name__ == '__main__':
    original_annotation_path = ("./captions_train2014.json")
    original_file_path = "/data/wangyan/COCO_Captions/train2014/"
    target_file_path = "coco_annotations_empty_lines.txt"
    target_file = open(target_file_path, mode='w')
    annotation_file = json.load(open(original_annotation_path))
    for x in annotation_file["annotations"]:
        image_id = x['image_id']
        caption = x['caption']
        image_name = ""
        for y in annotation_file["images"]:
            if y['id'] == image_id:
                image_name = y['file_name']
            if image_name != "":
                break
        target_file.writelines([image_name, '#', caption, '\n'])
    target_file.close()