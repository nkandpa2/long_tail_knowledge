# Large Language Models Struggle to Learn Long-Tail Knowledge

This repo contains the code and data used for the analysis in [Large Language Models Struggle to Learn Long-Tail Knowledge](https://arxiv.org/abs/2211.08411)

## Pre-training Dataset Entities
As part of this project, we ran the [DBPedia Spotlight](https://www.dbpedia-spotlight.org) entity linker on large-scale LM pre-training datasets such as The Pile, C4, ROOTS, OpenWebText, and Wikipedia. The entity linking data is hosted on [HuggingFace Hub](https://huggingface.co/datasets/nkandpa2/pretraining_entities). To get this data, you can either download it from their web UI, download it with Python using the instructions from [here](https://huggingface.co/docs/huggingface_hub/how-to-downstream), or run `git clone https://huggingface.co/datasets/nkandpa2/pretraining_entities` to clone the HuggingFace repo (note that this requires git lfs to be installed to actually download the data).

### Format
There are five files:
- `the_pile_entity_map.npz`
- `c4_entity_map.npz`
- `roots_entity_map.npz`
- `openwebtext_entity_map.npz`
- `wikipedia_entity_map.npz`

Each file can be loaded with `numpy.load` and is a dictionary mapping from DBPedia URI strings to numpy arrays of pre-training dataset indices where that entity occurs.

## QA Dataset Entities
To analyze the effect of pre-training entities on QA performance, we entity link Natural Questions and Trivia QA. The QA entity linking data is hosted on [HuggingFace Hub](https://huggingface.co/datasets/nkandpa2/qa_entities). To get this data, you can either download it from their web UI, download it with Python using the instructions from [here](https://huggingface.co/docs/huggingface_hub/how-to-downstream), or run `git clone https://huggingface.co/datasets/nkandpa2/qa_entities` to clone the HuggingFace repo (note that this requires git lfs to be installed to actually download the data).

### Format
There are four files:
- `nq_train_entities.jsonl`
- `nq_validation_entities.jsonl`
- `trivia_qa_unfiltered.nocontext_train_entities.jsonl`
- `trivia_qa_unfiltered.nocontext_validation_entities.jsonl`

The Natural Questions files are in the order of [nq_open](https://huggingface.co/datasets/nq_open) dataset and the Trivia QA files are in the order of the `unfiltered.nocontext` split of the [trivia_qa](https://huggingface.co/datasets/trivia_qa) dataset.

These are each jsonlines files. Each line in the file is a dictionary with the following structure:
```
{   
    'q_entities': <list of entities found in the question>,
    'a_entities': <list of entities found in the answer aliases>
}
```
The lists of entities are also dictionaries with the structure:
```
{
    'URI': <dbpedia entity URI>,
    'support': <dbpedia support>,
    'types': <dbpedia types>,
    'surfaceForm': <string in Q or A that was linked to this entity>,
    'offset': <location in Q or A where surface form is found>,
    'similarityScore': <dbpedia similarity score>,
    'percentageOfSecondRank': <dbpedia percentage of second ranked entity>
}
```
## Code
We're in the process of cleaning this up right now. Will push shortly. 
