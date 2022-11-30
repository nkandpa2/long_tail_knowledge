# Relevant Document Counting

## Heuristic
The heuristic used in the paper to estimate the number of relevant documents is as follows:
1. Find the most commonly found answer entity in the list of answer aliases
2. Find the question entity that co-occurs the most with the answer entity in the pre-training data
3. Return the co-occurrence count as the estimated number of relevant documents

To run this heuristic for a QA dataset and a pre-training dataset run the command: 
```console
python count_relevant_docs.py <path to qa entities .jsonl file> \
                              <path to pre-training entities .npz file> \
                              <output directory> \
                              --qa_split <qa dataset split used to generate entities> \
                              --type qa_co_occurrence
```
This will produce a json file called `qa_co_occurrence_split=<qa_split>.json` in the specified output directory. This file will just be a list of the number of relevant documents for each example in the QA dataset.

The script also supports just returning the count of the most common question or answer entity. That can by modifying the `--type` argument:
```console
python count_relevant_docs.py <path to qa entities .jsonl file> \
                              <path to pre-training entities .npz file> \
                              <output directory> \
                              --qa_split <qa dataset split used to generate entities> \
                              --type <q_occurrence | a_occurrence>
```
