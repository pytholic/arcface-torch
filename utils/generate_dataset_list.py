import os


def dataset_list(dataset_path, dataset_list):
    label_list = os.listdir(dataset_path)
    f = open(dataset_list, "w")
    k = 0
    for i in label_list:
        label_path = os.path.join(dataset_path, i)
        if os.listdir(label_path):
            image_list = os.listdir(label_path)
            for j in image_list:
                image_path = os.path.join(
                    i, j
                )  # Replace `i` with label_path for absolute path
                f.write(image_path + "  " + str(k) + "\n")
        k = k + 1
    f.close()


if __name__ == "__main__":
    dataset = "/Users/3i-a1-2021-15/Developer/projects/face-tracking/reid/arcface-pytorch/data/Datasets/webface/CASIA-maxpy-clean"
    list = "/Users/3i-a1-2021-15/Developer/projects/face-tracking/reid/arcface-pytorch/data/Datasets/webface/train_list.txt"
    dataset_list(dataset, list)
