

if __name__ == '__main__':
    original_file_path = "coco_annotations_abnormal.txt"
    target_file_path = "coco_annotations.txt"

    original_file = open(original_file_path, "r")
    target_file = open(target_file_path, 'w')

    for line in original_file.readlines():
        image_text = line.split("#")
        if image_text[0][-1] == "g" and image_text[1] != "":
            target_file.write(line)
    original_file.close()
    target_file.close()