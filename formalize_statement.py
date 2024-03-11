import argparse
from utils import read_jsonl, write_jsonl, inference_with_prompts
import random
import numpy as np
import copy
import os
random.seed(47)
np.random.seed(47)


def get_category(name, is_prompt):
    if is_prompt:
        if 'intermediate algebra' in name.lower():
            return 'intermediate_algebra'
        elif 'prealgebra' in name.lower():
            return 'prealgebra'
        elif 'algebra' in name.lower():
            return 'algebra'
        elif 'numbertheory' in name.lower() or 'number_theory' in name.lower() or 'number theory' in name.lower():
            return 'number_theory'
        elif 'gsm8k' in name.lower():
            return 'gsm8k'
        elif 'multiarith' in name.lower():
            return 'multiarith'
        else:
            return 'unknown'
    else:
        if 'intermediate algebra' in name.lower():
            return 'intermediate_algebra'
        elif 'prealgebra' in name.lower():
            return 'prealgebra'
        elif 'algebra' in name.lower():
            return 'algebra'
        elif 'number theory' in name.lower() or 'number_theory' in name.lower():
            return 'number_theory'
        elif 'gsm8k' in name.lower():
            return 'gsm8k'
        elif 'multiarith' in name.lower():
            return 'multiarith'
        else:
            return 'unknown'


def load_formalize_data(jsonl_path):
    json_data = read_jsonl(jsonl_path)
    for e in json_data:
        assert 'problem_name' in e and 'category' in e and 'generated_informal_statement' in e and 'generated_informal_proof' in e and 'short_answer' in e and 'short_prediction' in e
    return json_data


def get_all_combination_custom(tag, problem_type, informal_statement, all_available_prompts_d, n, max_num):
    assert len(all_available_prompts_d) >= n
    all_combinations = []
    needed_prompts = None
    if problem_type == 'number_theory' and 'base' in informal_statement:
        needed_prompts = []
        for k, v in all_available_prompts_d.items():
            if 'digits_in_base' in v['formal_statement']:
                needed_prompts.append(k)
    while len(all_combinations) < max_num:
        current_combination = list(np.random.choice(list(all_available_prompts_d.keys()), n, replace=False))
        if tag.split('|')[0] in current_combination:
            continue
        if needed_prompts is not None and len(set(needed_prompts).intersection(set(current_combination))) == 0:
            continue
        random.shuffle(current_combination)
        all_combinations.append(tuple(current_combination))
    return all_combinations


def get_single_sample_formatted(informal_statement, problem_template, prompt_prefix):
    assert not informal_statement.endswith('\n\nSolution:')
    assert not informal_statement.startswith('Problem:\n')
    total_prompt = prompt_prefix + '\n\n' + problem_template.format(informal_statement)
    return total_prompt


def get_all_samples_formatted(prompts_by_category, prefix_template, problem_template, informal_statement, problem_type, tag, max_num, n=3):
    # get all samples for one problem formatted
    # Sample n prompts and give their tags
    if problem_type not in prompts_by_category:
        prompts = {}
        for k in prompts_by_category:
            prompts.update({**prompts_by_category[k]})
    else:
        prompts = prompts_by_category[problem_type]
    # always permute prompts
    combinations = get_all_combination_custom(tag, problem_type, informal_statement, prompts, n, max_num)

    prompt_strings = []
    for e in combinations:
        processed_sampled_prompts = []
        for prompt_tag in e:
            sampled_prompt_data = prompts[prompt_tag]
            formatted_formal_statement = sampled_prompt_data['formal_statement']
            if type(prefix_template) is list:
                sampled_prompt = prefix_template[1].format(sampled_prompt_data['informal_statement'],
                                                        formatted_formal_statement)
            else:
                sampled_prompt = prefix_template.format(sampled_prompt_data['informal_statement'], formatted_formal_statement)
            processed_sampled_prompts.append(sampled_prompt)
        prompt_string = "\n\n".join(processed_sampled_prompts)
        if type(prefix_template) is list:
            prompt_string = prefix_template[0] + prompt_string
        prompt_strings.append(prompt_string)
    assert len(prompt_strings) == len(combinations)

    results = []
    for i in range(len(prompt_strings)):
        single_sample = get_single_sample_formatted(informal_statement, problem_template, prompt_strings[i])
        results.append([single_sample, combinations[i]])
    return results


def get_formalize_statement_prompts():
    prompts_by_category = {}
    all_prompt_examples_number_theory = read_jsonl('prompts/formal_statement/number_theory_formal_statement.jsonl')
    all_prompt_examples_algebra = read_jsonl('prompts/formal_statement/algebra_formal_statement.jsonl')
    all_prompt_examples_prealgebra = read_jsonl('prompts/formal_statement/prealgebra_formal_statement.jsonl')
    all_prompt_examples_gsm8k = read_jsonl('prompts/formal_statement/gsm8k_formal_statement.jsonl')
    all_prompt_examples_multiarith = read_jsonl('prompts/formal_statement/multiarith_formal_statement.jsonl')
    all_prompt_examples = all_prompt_examples_number_theory + all_prompt_examples_algebra + all_prompt_examples_prealgebra
    for e in all_prompt_examples:
        category = get_category(e['category'], is_prompt=True)
        if category not in prompts_by_category:
            prompts_by_category[category] = {}
        prompt_data = {'informal_statement': e['informal_statement'], 'formal_statement': e['formal_statement']}
        prompts_by_category[category][e['problem_name']] = prompt_data
    # gsm8k
    for e in all_prompt_examples_gsm8k:
        category = 'gsm8k'
        if category not in prompts_by_category:
            prompts_by_category[category] = {}
        prompt_data = {'informal_statement': e['informal_statement'], 'formal_statement': e['formal_statement']}
        prompts_by_category[category][e['problem_name']] = prompt_data
    # multiarith
    for e in all_prompt_examples_multiarith:
        category = 'multiarith'
        if category not in prompts_by_category:
            prompts_by_category[category] = {}
        prompt_data = {'informal_statement': e['informal_statement'], 'formal_statement': e['formal_statement']}
        prompts_by_category[category][e['problem_name']] = prompt_data
    return prompts_by_category


def formalize_statement_postprocess(all_samples):
    for i in range(len(all_samples)):
        all_samples[i]['formal_statement'] = all_samples[i]['formal_statement'].replace('\xa0', ' ').strip()


def main(args):
    jsonl_save_path = '{0}/formalize_statement.jsonl'.format(args.data_path.replace('informal_prove', 'formalize_statement').rsplit('/', 1)[0])
    inference_result_path = jsonl_save_path.replace('formalize_statement.jsonl', 'formalize_statement_after_inference.jsonl')
    final_save_path = inference_result_path.replace('formalize_statement_after_inference.jsonl', 'formalize_statement_done_processed.jsonl')
    prompts_by_category = get_formalize_statement_prompts()
    prefix_template = [
        "Translate the following natural language statement into an Isabelle formal statement. Note that the provided answer may not be correct. You should not try to correct it and please translate the statement as is.\n\nBelow are a few examples of such translations.\n\n",
        'Informal statement:\n(*###\n\n{0}\n\n###*)\n\nFormal statement:\n{1}']
    problem_template = 'Informal statement:\n(*###\n\n{0}\n\n###*)\n\nFormal statement:\n'
    stops = [args.stop]
    formalize_data = load_formalize_data(args.data_path)
    all_samples = []
    for i in range(len(formalize_data)):
        problem_name = formalize_data[i]['problem_name']
        informal_statement = formalize_data[i]['generated_informal_statement']
        problem_type = get_category(formalize_data[i]['category'], is_prompt=False)
        single_problem_samples = get_all_samples_formatted(prompts_by_category, prefix_template, problem_template, informal_statement, problem_type, problem_name, args.max_num_prompt_combination, n=args.num_shots)
        assert len(single_problem_samples) == args.max_num_prompt_combination
        for e in single_problem_samples:
            single_sample_data = copy.deepcopy(formalize_data[i])
            single_sample_data.update({'formalize_statement_prompt': e[0], 'formalize_statement_shots_names': e[1]})
            all_samples.append(single_sample_data)
    write_jsonl(all_samples, jsonl_save_path)
    print('writing {0} samples'.format(len(all_samples)))
    if os.path.exists(inference_result_path):
        all_samples = read_jsonl(inference_result_path)
    else:
        inference_with_prompts(all_samples, 'formalize_statement_prompt', 'formal_statement', inference_result_path, args.model_name, args.temperature, args.top_p, args.max_token, stops, args.awq)
    formalize_statement_postprocess(all_samples)
    write_jsonl(all_samples, final_save_path)
    print('done inference {0} samples'.format(len(all_samples)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-m', '--model_name', default='', type=str, help='')
    parser.add_argument('-d', '--data_path', default='', type=str, help='experiments/number_theory/informal_prove/prediction_filtered.jsonl')
    parser.add_argument('-max_num_prompt_combination', '--max_num_prompt_combination', default=10, type=int, help='')
    parser.add_argument('-num_shots', '--num_shots', default=10, type=int, help='')
    parser.add_argument('-t', '--temperature', default=0.3, type=float, help='')
    parser.add_argument('-top_p', '--top_p', default=0.95, type=float, help='')
    parser.add_argument('-max_token', '--max_token', default=400, type=int, help='')
    parser.add_argument('-stop', '--stop', default='Informal', type=str, help='')
    parser.add_argument('-awq', '--awq', default=0, type=int, help='autoawq')

    args = parser.parse_args()
    print(vars(args))
    main(args)
