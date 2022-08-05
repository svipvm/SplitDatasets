import os, json, random, shutil, sys

# convert coco to yolo formation
def coco_to_yolo(bbox, width,  height):
    x_min, y_min, ant_width, ant_height = bbox
    x_center = x_min + ant_width / 2.0
    y_center = y_min + ant_height / 2.0
    return x_center / width, y_center / height, ant_width / width, ant_height / height

# get all annotations of one image
def find_ants_and_convert(ant_dict, image_id, width, height):
    ants = []
    for item in ant_dict[image_id]:
        x_center, y_center, weight, height = coco_to_yolo(item['bbox'], width, height)
        ants.append({
            'x_center': x_center,
            'y_center': y_center,
            'height': height,
            'width': weight,
            'category_id': item['category_id']
        })
    return ants

def get_original_info(root_path, ant_file_name):
    ant_file_path = root_path + os.path.sep + 'Annotations' + os.path.sep + ant_file_name

    # read Coco-format data
    with open(ant_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # get annotations of each images
    ant_dict = {}
    for item in data['annotations']:
        if item['image_id'] not in ant_dict.keys():
            ant_dict[item['image_id']] = []
        ant_dict[item['image_id']].append({
            'bbox': item['bbox'],
            'category_id': item['category_id']
        })

    # get all annotations of all images
    original_list = []
    for item in data['images']:
        ants = find_ants_and_convert(ant_dict, item['id'], item['width'], item['height'])
        item_dict = {
            'id': item['id'],
            'filename': item['file_name'],
            'annotations': ants
        }
        original_list.append(item_dict)
    # for item in original_list:
    #     print(item)

    return original_list, data['image_nums'], data['categories']

# split data to (train, valid, test) for special rate
def split_data(original_list, split_rate):
    random.shuffle(original_list)
    num_datasets = len(original_list)
    split_count, split_datasets = 0, []
    for i in range(len(split_rate) - 1):
        count = int(num_datasets * split_rate[i])
        split_datasets.append(original_list[split_count: split_count + count])
        split_count += count
    split_datasets.append(original_list[split_count:])
    return split_datasets

# copy datasets to this path
def build_datasets(root_path, split_data_dict):
    original_images_root_path = root_path + os.path.sep + "Images" + os.path.sep
    for data_case, data_list in split_data_dict.items():
        images_path = root_path + os.path.sep + "Links" + os.path.sep
        labels_path = root_path + os.path.sep + "Labels" + os.path.sep
        os.makedirs(images_path, exist_ok=True)
        os.makedirs(labels_path, exist_ok=True)
        images_txt = images_path + data_case + '.txt'
        if os.path.exists(images_txt):
            with open(images_txt, 'w') as f:
                f.write('')
        for image_data in data_list:
            original_image_path = original_images_root_path + image_data['filename']
            # copy_image_path = images_path + image_data['filename']
            copy_label_path = labels_path + image_data['filename'].split('.')[0] + '.txt'
            # old function: copy image file
            # shutil.copyfile(original_image_path, copy_image_path)
            # new function: build txt about path
            with open(images_txt, 'a') as f:
                f.write(original_image_path + '\n')
            with open(copy_label_path, 'w') as f:
                for annotation in image_data['annotations']:
                    f.write("%d %f %f %f %f\n" % (
                        annotation['category_id'],
                        annotation['x_center'],
                        annotation['y_center'],
                        annotation['width'],
                        annotation['height']
                    ))
        print('path [ %s ]: [ %s ]' % (data_case, images_txt))
    # end

def memu():
    print("some command follow\n"
          "--h show help memu\n"
          "--source [dir] original images path\n"
          "--coco [file] Coco Annotation filename\n"
          # "--dataset [dataset] the name of dataset\n"
          "--split [name rate] the name and rate of split data")

if __name__ == '__main__':
    args = sys.argv
    if len(args) < 2:
        memu()
        exit(0)
    arg_name = ['--source', '--coco', '--dataset', '--split']
    root_path, ant_file_name, dataset, split_dict = None, None, None, {}
    index = 1
    while index < len(args):
        if arg_name[0] in args[index]:
            root_path = args[index + 1]
            index = index + 2
        elif arg_name[1] in args[index]:
            ant_file_name = args[index + 1]
            index = index + 2
        elif arg_name[2] in args[index]:
            dataset = args[index + 1]
            index = index + 2
        elif arg_name[3] in args[index]:
            split_dict[args[index + 1]] = float(args[index + 2])
            index = index + 3
        else:
            memu()
            exit(0)

    print('arguments information')
    print('    images root path:', root_path)
    # print('annotation file name:', ant_file_name)
    print('        dataset name:', dataset)
    print('          split info:', split_dict)

    # root_path = "D:\\datasets\\NippleDetection\\DatasetId\\"
    # ant_file_name = 'coco_info.json'
    original_list, image_count, categories = get_original_info(root_path, ant_file_name)

    # rate = [0.7, 0.2, 0.1] # train, valid, test
    datasets = split_data(original_list, [value for key, value in split_dict.items()])
    data_dict = {name: data for (name, value), data in zip(split_dict.items(), datasets)}
    # build_datasets(root_path, dataset, data_dict)
    build_datasets(root_path, data_dict)

    print('split data to success!')
    print('total of splitting data:',
          {name: len(data) for (name, _), data in zip(split_dict.items(), datasets)})
    print('        total of images:', image_count)
    print('             categories:', [category['name'] for category in categories])