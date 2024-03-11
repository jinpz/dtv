import json
import os
from vllm import LLM, SamplingParams
from collections import Counter
import numpy as np


def create_dir_if_necessary(file_path):
    file_dir = file_path.rsplit('/', 1)[0]
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)


def read_jsonl(path):
    results = []
    with open(path, 'r') as f:
        lines = f.readlines()
    for i in range(len(lines) - 1, -1, -1):
        results.insert(0, json.loads(lines[i]))
        del lines[i]
    # for line in lines:
    #     results.append(json.loads(line))
    return results


def write_jsonl(results, path):
    create_dir_if_necessary(path)
    with open(path, 'w') as f:
        f.write('\n'.join(json.dumps(e) for e in results))


def inference_with_prompts(jsonl_data, inference_attribute, save_attribute, save_path, model_name, temperature, top_p, max_token, stop, awq):
    prompts = [e[inference_attribute] for e in jsonl_data]
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_token, stop=stop)
    # hardcoded
    gpu_memory_utilization = 0.9
    download_dir = '/scratch/jz563/'
    if awq:
        llm = LLM(model=model_name, gpu_memory_utilization=gpu_memory_utilization, download_dir=download_dir, quantization='AWQ')
    else:
        llm = LLM(model=model_name, gpu_memory_utilization=gpu_memory_utilization, download_dir=download_dir)
    outputs = llm.generate(prompts, sampling_params)
    for i in range(len(outputs)):
        jsonl_data[i][save_attribute] = outputs[i].outputs[0].text
    write_jsonl(jsonl_data, save_path)
    return jsonl_data


def rekey_preprocess(raw_dataset, style):
    for e in raw_dataset:
        e['problem_name'] = e.pop('id')
        e['generated_informal_statement'] = e.pop('question')
        informal_statement = e['generated_informal_statement']
        if style == 'math':
            assert informal_statement.endswith('\n\nSolution:')
            assert informal_statement.startswith('Problem:\n')
            informal_statement = informal_statement[len('Problem:\n'):-len('\n\nSolution:')]
        elif style == 'gsm8k' or style == 'multiarith':
            assert informal_statement.endswith('\nA:')
            assert informal_statement.startswith('Q: ')
            informal_statement = informal_statement[len('Q: '):-len('\nA:')]
        # always append short prediction answer to the informal statement
        informal_statement += ' Show that it is {0}.'.format(e['short_prediction'])
        e['generated_informal_statement'] = informal_statement
        e['generated_informal_proof'] = e.pop('prediction')
        if style == 'gsm8k' or style == 'multiarith':
            if e['generated_informal_proof'].endswith('\n\nQ:'):
                e['generated_informal_proof'] = e['generated_informal_proof'][:-len('\n\nQ:')]
            assert e['category'] == ''
            e['category'] = style.upper()
        if 'full_prompt' in e.keys():
            e['informal_prove_prompt'] = e.pop('full_prompt')
        e['generated_informal_proof'] = e['generated_informal_proof'].strip()


def topk_majority_answers(predictions, topk=1):
    all_predictions = [e['short_prediction'] for e in predictions if e['short_prediction'] != '']
    votes = Counter(all_predictions).most_common()
    return [answer for answer, _ in votes[:topk]], Counter(all_predictions)


def majority_voting_tag(raw_dataset):
    # determine if majority voting for that question is correct and how many answers for a particular answer
    question_proofs_d = {}
    majority_vote_correct_questions = []
    for e in raw_dataset:
        question_number = e['problem_name'].split('|')[0].replace('problem', '')
        if question_number not in question_proofs_d:
            question_proofs_d[question_number] = []
        question_proofs_d[question_number].append(e)
    for question_number in question_proofs_d.keys():
        correct_answer = question_proofs_d[question_number][0]['short_answer']
        majority_prediction, votes = topk_majority_answers(question_proofs_d[question_number])
        if len(majority_prediction) == 0:  # all predictions are empty
            majority_vote_correct = -1
        else:
            if correct_answer == majority_prediction[0]:
                majority_vote_correct = 1
                majority_vote_correct_questions.append(question_number)
            else:
                majority_vote_correct = 0
        for e in question_proofs_d[question_number]:
            e['majority_vote_correct'] = majority_vote_correct
            if e['short_prediction'] in votes:
                e['short_prediction_count'] = votes[e['short_prediction']]
                if e['short_prediction'] == majority_prediction[0]:
                    e['prediction_is_majority_answer'] = 1
                else:
                    e['prediction_is_majority_answer'] = 0
            else:
                assert e['short_prediction'] == ''
                e['short_prediction_count'] = -1  # unknown
                e['prediction_is_majority_answer'] = 0
    return majority_vote_correct_questions


def upper_bound_acc(raw_dataset):
    # calculate by category
    problem_at_least_one_success_d = {}
    question_category_d = {}
    questions_proofs_d = {}
    majority_vote_success = []
    for e in raw_dataset:
        question_number = e['problem_name'].split('|')[0].replace('problem', '')
        if question_number not in questions_proofs_d:
            questions_proofs_d[question_number] = []
        questions_proofs_d[question_number].append(e)
        if question_number not in problem_at_least_one_success_d:
            problem_at_least_one_success_d[question_number] = 0
        if question_number not in question_category_d:
            question_category_d[question_number] = e['category']
        if e['short_answer'] == e['short_prediction']:
            problem_at_least_one_success_d[question_number] = 1
    for k, v in questions_proofs_d.items():
        if v[0]['short_answer'] in topk_majority_answers(v)[0]:
            majority_vote_success.append(v[0]['category'])
        # modification comment out above
        # np.random.seed(47)
        # if v[0]['short_answer'] == np.random.choice(v, 1)[0]['short_prediction']:
        #     majority_vote_success.append(v[0]['category'])
        # end modification
    majority_vote_success_counter = Counter(majority_vote_success)
    all_problems = [question_category_d[e] for e in problem_at_least_one_success_d]
    upper_bound_success = [question_category_d[k] for k in problem_at_least_one_success_d.keys() if problem_at_least_one_success_d[k] == 1]
    print('upper bound success {0} / {1}'.format(len(upper_bound_success), len(all_problems)))
    all_problems_counter = Counter(all_problems)
    upper_bound_success_counter = Counter(upper_bound_success)
    for k in all_problems_counter.keys():
        print('{2} category upper bound success: {0} / {1}'.format(upper_bound_success_counter[k], all_problems_counter[k], k))
    for k in majority_vote_success_counter:
        print('{0} category majority vote success: {1} / {2}'.format(k, majority_vote_success_counter[k], all_problems_counter[k]))


def further_filter_dataset(raw_dataset, drop_empty_prediction, drop_pure_problem, drop_no_correct_proof_problem, drop_incorrect_answer_proof, drop_asy):
    question_proofs_d = {}
    for e in raw_dataset:
        question_number = e['problem_name'].split('|')[0].replace('problem', '')
        if drop_asy and '[asy]' in e['generated_informal_statement']:
            continue
        if drop_empty_prediction and e['short_prediction'] == '':
            continue
        if drop_incorrect_answer_proof and e['short_prediction'] != e['short_answer']:
            continue
        if question_number not in question_proofs_d:
            question_proofs_d[question_number] = []
        question_proofs_d[question_number].append(e)
    results = []
    for question_number in question_proofs_d.keys():
        current_question_proofs = question_proofs_d[question_number]
        correct_answer = current_question_proofs[0]['short_answer']
        all_predictions = [e['short_prediction'] for e in current_question_proofs]
        if drop_pure_problem and len(np.unique(all_predictions)) == 1:
            continue
        if drop_no_correct_proof_problem and correct_answer not in all_predictions:
            continue
        results.extend(current_question_proofs)
    return results
