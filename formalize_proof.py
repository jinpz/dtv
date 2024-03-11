from utils import inference_with_prompts, read_jsonl, write_jsonl, json
import os
import copy
import numpy as np
import random


def get_the_type(problem_name):
    if "algebra" in problem_name.lower():
        return "algebra"
    elif "number_theory" in problem_name.lower() or 'numbertheory' in problem_name.lower() or 'number theory' in problem_name.lower():
        return "number_theory"
    else:
        return "unknown"


def get_formalize_proof_prompts():
    prompts_path = 'prompts/formal_solution/'
    prompts_by_category = {}
    for prompt_file in os.listdir(prompts_path):
        if prompt_file.endswith("json"):
            prompt_file_path = os.path.join(prompts_path, prompt_file)
            prompt_json = json.load(open(prompt_file_path))
            if 'formal_proof' not in prompt_json or 'informal_proof' not in prompt_json:
                continue
            tag = prompt_json["tag"]
            category = prompt_json["category"].lower()
            prompt_data = {"formal_statement": prompt_json['formal_statement'], 'formal_proof': prompt_json['formal_proof'], 'informal_statement': prompt_json['informal_statement'], 'informal_proof': prompt_json['informal_proof']}
            if category not in prompts_by_category:
                prompts_by_category[category] = {}
            prompts_by_category[category][tag] = prompt_data
    return prompts_by_category


def load_formalize_data(jsonl_path):
    json_data = read_jsonl(jsonl_path)
    for e in json_data:
        assert 'problem_name' in e and 'formal_statement' in e and 'generated_informal_statement' in e and 'generated_informal_proof' in e
    return json_data


def get_all_combination(tag, all_available_prompts_d, n, permute, max_num=-1):
    assert len(all_available_prompts_d) >= n
    assert permute
    all_combinations = []
    while len(all_combinations) < max_num:
        current_combination = list(np.random.choice(list(all_available_prompts_d.keys()), n, replace=False))
        if tag in current_combination:
            continue
        random.shuffle(current_combination)
        all_combinations.append(tuple(current_combination))
    return all_combinations


def extract_boxed_content_and_indices(proof_string: str):
    starting_index = proof_string.find("\\boxed{")
    opening_brackets = 0
    for i in range(starting_index+len("\\boxed{"), len(proof_string)):
        if proof_string[i] == "}":
            if opening_brackets == 0:
                return proof_string[starting_index+len("\\boxed{"):i], \
                        (starting_index, i)
            else:
                opening_brackets -= 1
        elif proof_string[i] == "{":
            opening_brackets += 1
        else:
            pass


def get_single_sample_formatted(informal_statement, problem_template, informal_proof, formal_statement, prompt_prefix, add_final_answer_informal_statement, delete_comments=False):
    if delete_comments:
        prompt_prefix_lines = [line.strip() for line in prompt_prefix.split("\n")]
        lines_to_delete = []
        to_delete = False
        for i, line in enumerate(prompt_prefix_lines):

            if line.startswith("(*"):
                assert not to_delete
                to_delete = True

            if to_delete:
                lines_to_delete.append(i)

            if line.endswith("*)"):
                assert to_delete
                to_delete = False
        assert not to_delete
        prompt_prefix_lines = [line for i, line in enumerate(prompt_prefix_lines) if i not in lines_to_delete]
        prompt_prefix = "\n".join(prompt_prefix_lines)

    if add_final_answer_informal_statement and "\\boxed{" in informal_proof:
        result = extract_boxed_content_and_indices(informal_proof)
        if result is None:
            pass
        else:
            content, (si, ei) = result
            content = content.strip()
            if "Show that it is" not in informal_statement:
                informal_statement = f"{informal_statement.strip()} Show that it is {content}."
            informal_proof = informal_proof[:si] + content + informal_proof[ei + 1:]

    total_prompt = prompt_prefix + '\n\n' + problem_template.format(formal_statement, informal_statement, informal_proof)
    return total_prompt


def get_all_samples_formatted(prompts_by_category, prefix_template, problem_template, informal_statement, informal_proof, formal_statement, problem_type, tag, max_num, permute, n=3, add_final_answer_informal_statement=False, delete_comments=False):
    # get all samples for one problem formatted
    # Sample n prompts and give their tags
    if problem_type not in prompts_by_category:
        prompts = {}
        for k in prompts_by_category:
            prompts.update({**prompts_by_category[k]})
    else:
        prompts = prompts_by_category[problem_type]
    combinations = get_all_combination(tag, prompts, n, permute, max_num)

    prompt_strings = []
    for e in combinations:
        processed_sampled_prompts = []
        for prompt_tag in e:
            sampled_prompt_data = prompts[prompt_tag]
            sampled_prompt = prefix_template.format(sampled_prompt_data['formal_statement'], sampled_prompt_data['informal_statement'], sampled_prompt_data['informal_proof'], sampled_prompt_data['formal_proof'])
            processed_sampled_prompts.append(sampled_prompt)
        prompt_string = "\n\n".join(processed_sampled_prompts)
        prompt_strings.append(prompt_string)

    assert len(prompt_strings) == len(combinations)
    results = []

    for i in range(len(combinations)):
        single_sample = get_single_sample_formatted(informal_statement, problem_template, informal_proof, formal_statement, prompt_strings[i], add_final_answer_informal_statement, delete_comments)
        results.append([single_sample, combinations[i]])
    return results


def get_prompt_template():
    prefix_template = 'Informal:\n(*### Problem\n\n{1}\n\n### Solution\n\n{2}*)\n\nFormal:\n{0}\n{3}'
    problem_template = 'Informal:\n(*### Problem\n\n{1}\n\n### Solution\n\n{2}*)\n\nFormal:\n{0}\n'
    split_word = 'Informal'
    return prefix_template, problem_template, split_word


def prepare_formalize_data(input_data_path, max_num_prompt_combination, num_shots, permute_shot, seed):
    random.seed(seed)
    np.random.seed(seed)
    prompts_by_category = get_formalize_proof_prompts()
    prefix_template, problem_template, _ = get_prompt_template()
    formalize_data = load_formalize_data(input_data_path)
    all_samples = []
    for i in range(len(formalize_data)):
        problem_name = formalize_data[i]['problem_name']
        informal_statement = formalize_data[i]['generated_informal_statement']
        informal_proof = formalize_data[i]['generated_informal_proof']
        formal_statement = formalize_data[i]['formal_statement']
        if 'category' in formalize_data[i]:
            problem_type = get_the_type(formalize_data[i]['category'])
        else:
            problem_type = get_the_type(problem_name)
        single_problem_samples = get_all_samples_formatted(prompts_by_category, prefix_template, problem_template, informal_statement, informal_proof, formal_statement, problem_type, problem_name, max_num_prompt_combination, permute_shot, n=num_shots)
        assert len(single_problem_samples) == max_num_prompt_combination
        for e in single_problem_samples:
            single_sample_data = copy.deepcopy(formalize_data[i])
            single_sample_data.update({'formalize_proof_prompt': e[0], 'formalize_shots_names': e[1]})
            all_samples.append(single_sample_data)
    return all_samples


def truncate_predictions(predictions):
    _, _, split_word = get_prompt_template()
    processed_predictions = []
    anomaly_counter = 0
    for prediction in predictions:
        prediction_split = prediction.split(split_word)
        if len(prediction_split) == 1:
            anomaly_counter += 1
        result = prediction_split[0].strip()
        # replace NBSP with space
        result = result.replace('\xa0', ' ')
        processed_predictions.append(result)
    return processed_predictions


def formalize_postprocess(predictions, formalize_json_data, save_path):
    processed_predictions = truncate_predictions(predictions)
    assert len(processed_predictions) % len(formalize_json_data) == 0
    num_decodes = len(predictions) // len(formalize_json_data)
    results = []
    for i in range(len(formalize_json_data)):
        for j in range(num_decodes):
            single_result = copy.deepcopy(formalize_json_data[i])
            single_result.update({'generated_formal_proof': processed_predictions[i*num_decodes + j]})
            results.append(single_result)
    write_jsonl(results, save_path)


if __name__ == '__main__':
    input_data_path = 'experiments/number_theory/formalize_statement/formalize_statement_done_processed.jsonl'
    prediction_save_path = 'experiments/number_theory/formalize/formalize_after_inference.jsonl'
    save_path = 'experiments/number_theory/formalize/formalize_done_processed.jsonl'
    max_num_prompt_combination = 1
    num_shots = 3
    permute_shot = True
    seed = 47
    all_samples = prepare_formalize_data(input_data_path, max_num_prompt_combination, num_shots, permute_shot, seed)
    predictions = inference_with_prompts(all_samples, 'foramlize_proof_prompt', 'formalize_proof', prediction_save_path, '', 0.6, 0.95, 1024, 'Informal', 0)
    formalize_postprocess(predictions, prediction_save_path, save_path)
