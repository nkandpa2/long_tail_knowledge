import numpy as np
import json
import pickle
import jsonlines
from tqdm.auto import tqdm
import argparse
import os
import utils

from country_entities import countries

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('qa_entities', help='path to QA entities file (jsonl)')
    parser.add_argument('training_entities', help='path to training entities file (npz)')
    parser.add_argument('output_dir', help='path to output directory')
    parser.add_argument('--qa_split', default='validation', help='split of qa dataset used')
    parser.add_argument('--type', default='qa_co_occurrence', choices=['qa_co_occurrence','q_occurrence','a_occurrence'], help='type of entity occurrence/co-occurrence to count')
    parser.add_argument('--save_examples', default=False, action='store_true', help='save examples where each relevant fact occurs')
    parser.add_argument('--exclude_countries', default=False, action='store_true', help='ignore countries in the list of question entities')

    args = parser.parse_args()
    return args

def count_occurrences(training_entities, entities):
    occurrence_count = 0
    for e in entities:
        examples = training_entities.get(e)
        if examples is None:
            continue
        count = len(examples)
        occurrence_count = max(occurrence_count, count)

    return occurrence_count

def count_co_occurrences(training_entities, q_entities, a_entities, common_a_entity):
    co_occurrence_count = 0
    co_occurrence_examples = []
    qa_pair = None
    if common_a_entity is None:
        return co_occurrence_count, co_occurrence_examples, qa_pair
    for a in common_a_entity:
        a_examples = training_entities.get(a)
        for q in q_entities:
            q_examples = training_entities.get(q)
            if q_examples is None or a_examples is None:
                continue
            examples = utils.intersect(a_examples, q_examples)
            count = len(examples)
            if count > co_occurrence_count:
                co_occurrence_count = count
                co_occurrence_examples = examples
                qa_pair = (q,a)

    return co_occurrence_count, co_occurrence_examples, qa_pair


def main(args):
    print('Loading training entities')
    training_entities = np.load(args.training_entities)
    
    print('Sorting training entity lists')
    training_entities = utils.sort_entity_map(training_entities)

    print('Counting occurrences')
    occurrences = []
    examples = []
    qa_pairs = []
    with open(args.qa_entities, 'r') as f:
        with jsonlines.Reader(f) as reader:
            for i, qa_example in enumerate(tqdm(reader)):
                q_entities = set([q['URI'] for q in qa_example['q_entities']])
                if args.exclude_countries:
                    q_entities -= countries
                a_entity_list = [a['URI'] for a in qa_example['a_entities'] if not a['URI'] in q_entities]
                a_entities = set(a_entity_list)
                a_entity_counts = {a:a_entity_list.count(a) for a in a_entities}
                if len(a_entity_counts) == 0:
                    common_a_entity = None
                else:
                    max_count = max(a_entity_counts.values())
                    common_a_entity = [a for a in a_entity_counts.keys() if a_entity_counts[a] == max_count]
                
                if args.type == 'qa_co_occurrence':
                    fact_occurrences, fact_examples, qa_pair = count_co_occurrences(training_entities, 
                                                                                      q_entities, 
                                                                                      a_entities, 
                                                                                      common_a_entity)
                    occurrences.append(fact_occurrences)
                    examples.append(fact_examples)
                    qa_pairs.append(qa_pair)

                elif args.type == 'q_occurrence':
                    question_occurrences = count_occurrences(training_entities, q_entities)
                    occurrences.append(question_occurrences)

                elif args.type == 'a_occurrence':
                    answer_occurrences = count_occurrences(training_entities, a_entities)
                    occurrences.append(answer_occurrences)


    fname = f'{args.type}_split={args.qa_split}.json'
    with open(os.path.join(args.output_dir, fname), 'w') as f:
        json.dump(occurrences, f)
    
    if args.save_examples:
        fname = f'{args.type}_examples_split={args.qa_split}.pkl'
        with open(os.path.join(args.output_dir, fname), 'wb') as f:
            pickle.dump({'examples': examples, 'qa_pairs': qa_pairs}, f)

if __name__ == '__main__':
    args = parse_args()
    main(args)
