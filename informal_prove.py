import argparse
from utils import read_jsonl, write_jsonl, inference_with_prompts, rekey_preprocess, upper_bound_acc, majority_voting_tag, further_filter_dataset
import re
import os


list_of_subs = [('an ', ''), ('a ', ''), ('.$', '$'), ('\\$', ''), (r'\ ', ''),
                (' ', ''), ('mbox', 'text'), (',\\text{and}', ','),
                ('\\text{and}', ','), ('\\text{m}', '\\text{}')]
list_of_words = [
    'square', 'ways', 'integers', 'dollars', 'mph', 'inches', 'ft', 'hours',
    'km', 'units', '\\ldots', 'sue', 'points', 'feet', 'minutes', 'digits',
    'cents', 'degrees', 'cm', 'gm', 'pounds', 'meters', 'meals', 'edges',
    'students', 'childrentickets', 'multiples'
]
list_of_words += [
    '\\text{s}', '\\text{.}', '\\text{\ns}', '\\text{}^2', '\\text{}^3',
    '\\text{\n}', '\\text{}', r'\mathrm{th}', r'^\circ', r'^{\circ}', r'\;',
    r',\!', '{,}', '"', '\\dots'
]


def format_solution(short_answer: str) -> str:
    """Formats answer for uniformization purposes."""
    short_answer = short_answer.strip()
    for el1, el2 in list_of_subs:
        short_answer = short_answer.replace(el1, el2)
    for el in list_of_words:
        short_answer = short_answer.replace(el, '')
    #
    short_answer = re.sub(r'(.*?)(\$)(.*?)(\$)(.*)', '$\\3$', short_answer)
    short_answer = re.sub(r'(\\text\{)(.*?)(\})', '\\2', short_answer)
    short_answer = re.sub(r'(\\textbf\{)(.*?)(\})', '\\2', short_answer)
    short_answer = re.sub(r'(\\overline\{)(.*?)(\})', '\\2', short_answer)

    # Not greedy
    short_answer = re.sub(r'(\\boxed\{)(.*)(\})', '\\2', short_answer)
    # \fracab=\frac{a}{b}, \frac{abc}{bef}=\frac{abc}{bef}, \fracabc=\frac{a}{b}c
    short_answer = re.sub(r'(frac)([^{])(.)', 'frac{\\2}{\\3}', short_answer)
    # \sqrta=\sqrt{a}, \sqrtab=sqrt{a}b
    short_answer = re.sub(r'(sqrt)([^{])', 'sqrt{\\2}', short_answer)

    short_answer = short_answer.split('=')[-1]

    short_answer = short_answer.replace('$', '')
    # Ie 100,000 -> 100000 , ...
    if short_answer.replace(',', '').isdigit():
        short_answer = short_answer.replace(',', '')
    return short_answer


def extract_short_predictions(jsonl_data):
    empty_prediction_count = 0
    for i in range(len(jsonl_data)):
        prediction = jsonl_data[i]['prediction']
        if 'The final answer is' not in prediction:
            jsonl_data[i].update({'short_prediction': ''})
            empty_prediction_count += 1
        else:
            short_prediction = format_solution(prediction.split('The final answer is')[-1])
            jsonl_data[i].update({'short_prediction': short_prediction})
    print('empty prediction count {0} out of {1}'.format(empty_prediction_count, len(jsonl_data)))


def main(args):
    jsonl_data = read_jsonl(args.data_path)
    save_path = args.save_dir + 'informal_prove/prediction_raw.jsonl'
    final_save_path = args.save_dir + 'informal_prove/prediction_filtered.jsonl'
    stops = [args.stop]
    if os.path.exists(save_path):
        jsonl_data = read_jsonl(save_path)
    else:
        inference_with_prompts(jsonl_data, 'full_prompt', 'prediction', save_path, args.model_name, args.temperature, args.top_p, args.max_token, stops, args.awq)

    extract_short_predictions(jsonl_data)
    write_jsonl(jsonl_data, save_path)
    # post processing
    rekey_preprocess(jsonl_data, 'math')
    majority_vote_correct_questions = majority_voting_tag(jsonl_data)
    upper_bound_acc(jsonl_data)
    cleaned_evaluation_dataset = further_filter_dataset(jsonl_data, True,
                                                        False, True,
                                                        False, True)

    write_jsonl(cleaned_evaluation_dataset, final_save_path)
    print('correct problem names', sorted(majority_vote_correct_questions))
    print('evaluation dataset length', len(cleaned_evaluation_dataset))
    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-m', '--model_name', default='', type=str, help='')
    parser.add_argument('-d', '--data_path', default='experiments/number_theory/math_number_theory.jsonl', type=str, help='')
    parser.add_argument('-s', '--save_dir', default='experiments/', type=str, help='')
    parser.add_argument('-t', '--temperature', default=0.6, type=float, help='')
    parser.add_argument('-top_p', '--top_p', default=0.95, type=float, help='')
    parser.add_argument('-max_token', '--max_token', default=2048, type=int, help='')
    parser.add_argument('-stop', '--stop', default='Problem:', type=str, help='')
    parser.add_argument('-awq', '--awq', default=0, type=int, help='autoawq')

    args = parser.parse_args()
    print(vars(args))
    main(args)
