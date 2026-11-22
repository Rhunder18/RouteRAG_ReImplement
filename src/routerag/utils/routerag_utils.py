import re

SYSTEM_PROMPT = """You are a biomedical expert answering medical and clinical questions based on scientific literature.
You must conduct reasoning inside <think> and </think> first every time you receive new information. After reasoning, if you need more biomedical knowledge or evidence from the literature, call the search engine using this strict format:
1. Retrieval Mode:
   - Use [passage] for semantic similarity-based dense retrieval over medical texts.
   - Use [graph] for structured knowledge-graph-based retrieval (entities, biological relations, and facts).
   - Use [graph][passage] to combine both retrieval methods.
2. Formulate your search query with specific biomedical keywords or entities.
3. Wrap the mode and query inside <search> and </search> tags.
Examples:
- <search> [passage] mitochondria role in lace plant programmed cell death </search>
- <search> [graph] lace plant programmed cell death </search>
- <search> [graph][passage] lace plant leaf remodeling PCD </search>
Retrieved evidence will be returned inside <information> and </information> tags. You can search iteratively as needed.
Once you have sufficient evidence, reason inside <think>...</think> whether the evidence supports, refutes, or is inconclusive regarding the question. Then output the final decision strictly inside <answer> and </answer> using ONLY ONE of the following three options:
- <answer> yes </answer> : The literature/evidence clearly supports the hypothesis or question.
- <answer> no </answer> : The literature/evidence refutes or does not support the hypothesis.
- <answer> maybe </answer> : The findings are mixed, uncertain, limited, or inconclusive.
Do not include any extra words or explanation inside <answer> and </answer>."""

STOP_STRINGS = ["</search>", "</answer>"]
RETHINK_MSG = f"\nMy previous action is invalid. If I want to search, I should put the query between <search> and </search>. If I want to give the final answer, I should put the answer between <answer> and </answer>.\n"

def extract_final_label(output_text):
    match = re.search(r"<answer>\s*(yes|no|maybe)\s*</answer>", output_text, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return "unknown"

def exact_match(prompts, completions, final_decision, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_final_label(r) for r in responses]
    return [1.0 if r == a else 0.0 for r, a in zip(extracted_responses, final_decision)]

def exact_match_with_efficency(prompts, completions, search_times, final_decision, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_final_label(r) for r in responses]
    percentile_95 = np.percentile(search_times, 95)
    scaled_times = []
    for t in search_times:
        scaled_time = t
        scaled_time = scaled_time * 0.5 / percentile_95
        scaled_time = min(scaled_time, 0.5)
        scaled_times.append(scaled_time)
    avg_scaled_time = sum(scaled_times) / len(scaled_times)
    rewards = []
    for r, a, st in zip(extracted_responses, final_decision, scaled_times):
        if r == a:
            efficiency_score = avg_scaled_time - st
            rewards.append(max(0.0, 1.0 + efficiency_score))
        else:
            rewards.append(0.0)
    return rewards