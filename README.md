# Large Language Models Struggle to Learn Long-Tail Knowledge

This repo contains the code and data used for the analysis in [Large Language Models Struggle to Learn Long-Tail Knowledge](https://arxiv.org/abs/2211.08411)

## Pre-training Dataset Entities
As part of this project, we ran the [DBPedia Spotlight](https://www.dbpedia-spotlight.org) entity linker on large-scale LM pre-training datasets such as The Pile, C4, ROOTS, OpenWebText, and Wikipedia. The entity linking data is hosted on [HuggingFace Hub](https://huggingface.co/datasets/nkandpa2/pretraining_entities). To get this data, you can either download it from their web UI, with the instructions from [here](https://huggingface.co/docs/huggingface_hub/how-to-downstream), or using git to clone the repo from HuggingFace.

### Format
There are five files:
- `the_pile_entity_map.npz`
- `c4_entity_map.npz`
- `roots_entity_map.npz`
- `openwebtext_entity_map.npz`
- `wikipedia_entity_map.npz`

Each file can be loaded with `numpy.load` and is a dictionary mapping from DBPedia URI strings to numpy arrays of pre-training dataset indices where that entity occurs.

## QA Dataset Entities
Need to push

## Code
We're in the process of cleaning this up right now. Will push shortly.
