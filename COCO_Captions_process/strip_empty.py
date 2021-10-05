
if __name__ == '__main__':
    original_file_path = "coco_annotations_empty_lines.txt"
    target_file_path = "coco_annotations_abnormal.txt"

    original_file = open(original_file_path, "r")
    target_file = open(target_file_path, 'w')

    for line in original_file.readlines():
        if line == '\n':
            line = line.strip('\n')
        target_file.write(line)
    original_file.close()
    target_file.close()