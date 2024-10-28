import json
import re
from sklearn.metrics import accuracy_score
import argparse
import os


#### object classification ####
def eval_cls_task(predictions, answers):
    """
    @func: evaluate [classify] task
    @task: image + instruction >>> categroy
    @metric: accuracy 
    """
    accuracy = accuracy_score(answers, predictions)
    return accuracy


#### referring expression generation ####
def eval_reg_task(predictions, answers):
    """
    @func: evaluate [REG] task
    @task: image + cooridinate >>> categroy
    @metric: accuracy 
    """
    accuracy = accuracy_score(answers, predictions)
    return accuracy


#### referring expression comprehension ####
# reference：https://github.com/facebookresearch/Detectron/blob/05d04d3a024f0991339de45872d02f2f50669b3d/lib/datasets/voc_eval.py#L54
def compute_tp_fp(pred_bbox, ans_bbox, ovthresh=0.5):
    tp = 0
    fp = 0

    iou = compute_iou(pred_bbox, ans_bbox)
    if iou > ovthresh:
        tp = 1
    else:
        fp = 1
    return tp, fp


# iou
# reference: https://github.com/facebookresearch/Detectron/blob/05d04d3a024f0991339de45872d02f2f50669b3d/lib/datasets/voc_eval.py#L185
def compute_iou(bbox1, bbox2):
    if len(bbox1) != 4 or len(bbox2) != 4:
        return 0

    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2

    inter_xmin = max(xmin1, xmin2)
    inter_ymin = max(ymin1, ymin2)
    inter_xmax = min(xmax1, xmax2)
    inter_ymax = min(ymax1, ymax2)

    inter_width = max(0, inter_xmax - inter_xmin + 1)
    inter_height = max(0, inter_ymax - inter_ymin + 1)
    if inter_width == 0 or inter_height == 0:
        return 0

    inter_area = inter_width * inter_height
    bbox1_area = (xmax1 - xmin1 + 1) * (ymax1 - ymin1 + 1)
    bbox2_area = (xmax2 - xmin2 + 1) * (ymax2 - ymin2 + 1)

    iou = inter_area / float(bbox1_area + bbox2_area - inter_area)
    return iou


def eval_rec_task(predictions, answers):
    """
    @func: evaluate [REC] task
    @task: image + instruction >>> categroy
    @metric: iou, precision, recall
    """
    ious = []
    precisions = []
    recalls = []
    for pred, ans in zip(predictions, answers):
        pred_bbox = list(map(int, re.findall(r'<\s*(\d+)\s*>', pred)))
        ans_bbox = list(map(int, re.findall(r'<\s*(\d+)\s*>', ans)))

        iou = compute_iou(pred_bbox, ans_bbox)
        ious.append(iou)

        # precision & recall
        tp, fp = compute_tp_fp(pred_bbox, ans_bbox)

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / len(ans_bbox) if len(ans_bbox) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)

    avg_iou = sum(ious) / len(ious)
    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    # ap = compute_ap(precisions, recalls)

    return {
        'avg_iou': avg_iou,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall
    }


# Check the number of predictions
def check_json_count(file_path, expected_count):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    count = len(data)
    if count != expected_count:
        raise ValueError(f"Incorrect, the expected number is {expected_count}, the actual number is {count}")
    return data


# evaluation
def eval_engine(args):

    try:
        if args.eval_mode == 'val':
            if args.eval_task == 'CLS' or args.eval_task == 'CAP':
                expected_count = 8929
            elif args.eval_task == 'REG' or args.eval_task == 'REC':
                expected_count = 4874
        elif args.eval_mode == 'test':
            if args.eval_task == 'CLS' or args.eval_task == 'CAP':
                expected_count = 45284
            elif args.eval_task == 'REG' or args.eval_task == 'REC':
                expected_count = 37631
        else:
            raise ValueError(f"Unknown mode type: {args.eval_mode}")

        data = check_json_count(args.json_file, expected_count)

    except ValueError as e:
        print(e)
        return

    predictions = []
    gpts = []
    for item in data:
        for conversation in item['conversations']:
            if conversation['from'] == 'prediction':
                prediction = conversation['value'].lower().replace('<|endoftext|>', '')
                predictions.append(prediction)
            elif conversation['from'] == 'gpt':
                gpt = conversation['value'].lower()
                gpts.append(gpt)

    accuracies = {}
    if args.eval_task == 'CLS':
        accuracies['CLS'] = eval_cls_task(predictions, gpts)
    elif args.eval_task == 'REC':
        refer_results = eval_rec_task(predictions, gpts)
        accuracies['REC'] = refer_results
    elif args.eval_task == 'REG':
        accuracies['REG'] = eval_reg_task(predictions, gpts)
    elif args.eval_task == 'CAP':
        print('Still working on it')

    if args.write_to_json:
        with open(args.json_file, "a") as f:
            f.write(",\n")
            json.dump({"eval_result": accuracies}, f, ensure_ascii=False, indent=4)
    else:
        txt_file_path = args.json_file.replace('.json', '_eval.txt')
        with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
            json.dump({"eval_result": accuracies}, txt_file, ensure_ascii=False, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", help="Path to the predicted json file.")
    parser.add_argument("--eval_task", help="The evaluation task")
    parser.add_argument("--eval_mode", help="The evaluation mode ('val' or 'test').")
    parser.add_argument("--write_to_json", type=bool, default=False, help="True/False means write eval result to JSON/TXT file")
    args = parser.parse_args()

    eval_engine(args)


if __name__ == "__main__":
    main()