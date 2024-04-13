import os
import editdistance

from utils import load_jsonl, load_pickle, extract_identifiers


from fuzzywuzzy import fuzz

debug = False

def evaluate_id_match_EM_F1(datas):
    em_list = []
    f1_list = []
    for data in datas:
        try:
            predict = data['predict'].strip()
        except:
            predict = data['choices'][0]['text'].split("\n")[0].strip()

        try:
            gt = data["groundtruth"].strip()
        except:
            gt = data['metadata']['ground_truth'].strip()

        pred_ids = list(extract_identifiers(predict))
        target_ids = list(extract_identifiers(gt))

        tp = 0
        fp = 0
        fn = 0
        for pid in pred_ids:
            if pid in target_ids:
                tp += 1
            else:
                fp += 1
        for tid in target_ids:
            if tid not in pred_ids:
                fn += 1

        em_list.append(int(pred_ids == target_ids))
        # precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        # recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1_list.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0)
    em = round(sum(em_list) / len(datas) * 100, 2)
    f1 = round(sum(f1_list) / len(datas) * 100, 2)
    return em, f1


def evaluate_line_EM_ES(data):
    em_scores = []
    es_scores = []
    for d in data:
        try:
            predict = d['predict'].strip()
        except:
            predict = d['choices'][0]['text'].split("\n")[0].strip()

        try:
            gt = d["groundtruth"].strip()
        except:
            gt = d['metadata']['ground_truth'].strip()
            # gt = d['metadata']['ground_truth']

        if gt.startswith(predict) and gt[len(predict):].strip().startswith("#"):
            gt = predict

        if gt == predict:
            em_scores.append(1)
        else:
            em_scores.append(0)

        es_scores.append(fuzz.ratio(predict, gt))
        # es_scores.append(1 - (editdistance.eval(predict, gt) / max(len(predict), len(gt))))

    print(f"{sum(em_scores)} / {len(data)}")
    avg_em_score = round(sum(em_scores)/len(em_scores), 4)
    avg_es_score = round(sum(es_scores)/len(es_scores), 2)
    return avg_em_score, avg_es_score

def evaluate(output_file):
    if not os.path.exists(output_file):
        return 0, 0, 0, 0

    if output_file.endswith(".jsonl"):
        data = load_jsonl(output_file)
    else:
        data = load_pickle(output_file)

    if len(data) == 0:
        return 0, 0, 0, 0

    line_em_score, line_es_score = evaluate_line_EM_ES(data)
    id_em_score, id_f1_score = evaluate_id_match_EM_F1(data)

    print(f"Line Exact Match: {line_em_score}")
    print(f"Line Edit Similarity: {line_es_score}")
    print(f"Identifier Exact Match: {id_em_score}")
    print(f"Identifier Match F1 Score: {id_f1_score}")
    return id_em_score, id_f1_score, line_em_score, line_es_score


if __name__ == "__main__":
    file_dir = 'data/generate/study/'
    for file in os.listdir(file_dir):
        if os.path.isfile(os.path.join(file_dir, file)):
            print('-' * 5 + file + '-' * 5)
            # data = load_jsonl(os.path.join(file_dir, file))
            # if len(data) != 2075:
            #     print(f'{file}: {len(data)}')
            evaluate(os.path.join(file_dir, file))

    # path = 'data/generate/study/java/codellama-sls.jsonl'
    # print(path)
    # evaluate(path)
