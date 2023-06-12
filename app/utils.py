import jieba
import os
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def segmentation(text_input):
    return jieba.lcut(text_input)

def read_class_file(path):
    """读取class文件"""
    with open(path, 'r', encoding='utf-8') as file:
        lable_lis = [label.strip() for label in file.readlines() if label.strip()]
        label_2_id = {label: i for i, label in enumerate(lable_lis)}
        id_2_label = {id: label for label, id in label_2_id.items()}
        return label_2_id, id_2_label