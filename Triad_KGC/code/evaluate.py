import csv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from openai import OpenAI
from tqdm import tqdm

from Triad_KGC.code.config import settings


class OpenAIEmbedding:
    def __init__(self, api_key: str, model: str = "text-embedding-3-small", base_url: str = "https://api.rcouyi.com/v1"):
        client_args = {"api_key": api_key}
        if base_url:
            client_args["base_url"] = base_url
        self.client = OpenAI(**client_args)
        self.model = model

    def embed(self, texts, **kwargs):
        response = self.client.embeddings.create(
            input=texts,
            model=self.model,
            **kwargs
        )
        return [np.array(item.embedding) for item in response.data]


class AnswerMatcher:
    def __init__(self, threshold: float):
        self.threshold = threshold
        self.embed_client = OpenAIEmbedding(api_key=settings.API_KEY,
                                            model=settings.EMBEDDING_MODEL,
                                            base_url=settings.BASE_URL)

    def standardize_text(self, text: str) -> str:
        if not text:
            return ""
        t = text.lower().strip()
        t = re.sub(r'[^一-龥\w\s]', '', t)
        return re.sub(r'\s+', ' ', t)

    def semantic_similarity(self, pred: str, true: str):
        p_std = self.standardize_text(pred)
        t_std = self.standardize_text(true)
        if p_std and t_std:
            emb_p, emb_t = self.embed_client.embed([p_std, t_std])
            if emb_p.ndim == 1:
                emb_p = emb_p.reshape(1, -1)
            if emb_t.ndim == 1:
                emb_t = emb_t.reshape(1, -1)
            sim = cosine_similarity(emb_p, emb_t)[0][0]
        else:
            sim = 0
        return sim

    def calculate_metrics_parallel(self, data, original_rows, max_workers=8):
        """
        Compute MRR, Hits@1, Hits@3 using multithreading.
        """
        if isinstance(data, dict):
            sample_list = []
            for key, value in data.items():
                if isinstance(key, tuple) and isinstance(value, dict):
                    for gold, preds in value.items():
                        sample_list.append((gold, preds))
                else:
                    sample_list.append((key, value))
        else:
            sample_list = data

        mrr_scores = []
        hits1 = hits3 = 0
        failed_indices = []

        def process_sample(true, preds, idx):
            # Exact match
            exact_rank = None
            for i, p in enumerate(preds, start=1):
                if p.strip().lower() == true.strip().lower():
                    exact_rank = i
                    break
            if exact_rank:
                return idx, 1.0 / exact_rank, int(exact_rank == 1), 1

            # Semantic similarity
            found = False
            rank = 0
            sims = [self.semantic_similarity(p, true) for p in preds[:3]]
            for i, sim in enumerate(sims, start=1):
                if sim >= self.threshold:
                    found = True
                    rank = i
                    break

            if not found:
                failed_indices.append(idx)

            return idx, 1.0 / rank if found else 0.0, int(rank == 1), int(found)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_sample, true, preds, idx) for idx, (true, preds) in enumerate(sample_list)]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Parallel Metrics", ncols=80):
                idx, mrr, h1, h3 = future.result()
                mrr_scores.append(mrr)
                hits1 += h1
                hits3 += h3

        total = len(sample_list)
        mrr = np.mean(mrr_scores) if mrr_scores else 0.0
        return mrr, hits1 / total, hits3 / total, [original_rows[i] for i in failed_indices] if original_rows else []


def parse_triple(triple_str):
    """
    Parse a triple string, return entity1, relation, entity2.
    Example: (A, r, B) -> ['A', 'r', 'B']
    """
    triple_str = triple_str.strip()
    triple_str = triple_str.strip('()')
    parts = [x.strip() for x in triple_str.split(',')]
    return parts if len(parts) == 3 else []


def split_triples(s):
    """
    Use regex to split all (xxx, xxx, xxx) segments where only two commas are inside the parentheses.
    """
    raw = re.findall(r'\([^\(\)]*?\)', s)
    return [t for t in raw if t.count(',') == 2]

def clean_entity_name(name: str) -> str:
    """
    Remove suffix like _XX_number from an entity name, e.g., "dog_NN_1" => "dog"
    """
    return re.sub(r'_[A-Z]{2,3}_\d+$', '', name)

def csv_to_eval_dict(csv_path):
    """
    Read CSV and return a dictionary {gold_entity: [predicted_entities]},
    mask_position determines whether to take the 0th or 2nd element of the triple.
    """
    original_rows = []
    eval_dict = {}
    count = 0
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mask_pos = row['Mask Position'].strip().lower()
            count += 1
            mask_idx = 0 if mask_pos == 'head' else 2

            # Parse gold triple
            gold_triples = split_triples(row['Gold Triple'].strip())
            if not gold_triples:
                continue
            gold_parts = parse_triple(gold_triples[0])
            if not gold_parts:
                continue
            gold_entity = gold_parts[mask_idx]
            gold_triple = tuple(gold_parts)

            # Parse predicted triples
            pred_entities = []
            for inst in split_triples(row['Completed Triples'].strip()):
                parts = parse_triple(inst)
                if parts:
                    pred_entities.append(clean_entity_name(parts[mask_idx]))

            if pred_entities:
                eval_dict[gold_triple] = {clean_entity_name(gold_entity): pred_entities}
                original_rows.append(row)
            else:
                pred_entities = ["None"]
                eval_dict[gold_triple] = {clean_entity_name(gold_entity): pred_entities}
                original_rows.append(row)

    return eval_dict, original_rows


if __name__ == '__main__':
    matcher = AnswerMatcher(threshold=0.8)
    eval_csv_path = ""
    eval_dict, original_rows = csv_to_eval_dict(eval_csv_path)
    mrr, h1, h3, failed_rows = matcher.calculate_metrics_parallel(data=eval_dict,
                                                                  original_rows=original_rows,
                                                                  max_workers=500)

    print(f"MRR: {mrr:.4f}, Hits@1: {h1:.4f}, Hits@3: {h3:.4f}")
