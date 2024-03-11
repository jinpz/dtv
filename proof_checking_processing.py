from utils import read_jsonl, write_jsonl
import numpy as np
import random
import argparse


def load_proof_checking_data(jsonl_path):
    json_data = read_jsonl(jsonl_path)
    for e in json_data:
        assert 'problem_name' in e and 'formal_statement' in e and 'generated_formal_proof' in e
    return json_data


def prepare_proof_checking_data(input_data_path, chunk_save_path, proofs_per_instance, shuffle, seed):
    random.seed(seed)
    proof_checking_data = load_proof_checking_data(input_data_path)
    if proofs_per_instance != -1:
        num_instances = int(np.ceil(len(proof_checking_data) / proofs_per_instance))
    else:
        proofs_per_instance = len(proof_checking_data)
        num_instances = 1
    if 'match_index' not in proof_checking_data[0]:
        print('adding match index')
        for i in range(len(proof_checking_data)):
            proof_checking_data[i].update({'match_index': i})
    else:
        print('match index already added')
    if shuffle:
        random.shuffle(proof_checking_data)
    for i in range(num_instances):
        current_assignment = proof_checking_data[i * proofs_per_instance:(i + 1) * proofs_per_instance]
        save_path = chunk_save_path + '/responses_chunks/' + 'responses_chunk_{0}.json'.format(i)
        write_jsonl(current_assignment, save_path)
    print('end index should be {0}'.format(num_instances))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='experiments/number_theory/complete_after_formalization.jsonl')
    parser.add_argument('--proofs_per_instance', type=int, default=-1)
    parser.add_argument('--shuffle', type=int, default=0)
    parser.add_argument('--seed', type=int, default=47)
    args = parser.parse_args()
    print(vars(args))

    proofs_per_instance = args.proofs_per_instance  # chunk size
    shuffle = args.shuffle
    seed = args.seed
    prepare_proof_checking_data(args.data_path, args.data_path.rsplit('/', 1)[0], proofs_per_instance, shuffle, seed)
