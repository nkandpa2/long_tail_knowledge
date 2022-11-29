# Entity Linking Scripts

## DBPedia Spotlight Entity Linker
The scripts in this directory use the [DBPedia Spotlight](https://www.dbpedia-spotlight.org) entity linker. The simplest way to run DBPedia Spotlight is to spin up their provided docker image that runs the DBPedia service on a specified port.

To run their English entity linker, you can use the following command:
`sudo docker run -tid --restart unless-stopped --name dbpedia-spotlight.en --mount source=spotlight-model,target=/opt/spotlight -p 2222:80 dbpedia/dbpedia-spotlight spotlight.sh en`
This will start a container with the name `dbpedia-spotlight.en` running the DBPedia service on port `2222`.

For entity linking large pre-training datasets it's necessary to start many instances of DBPedia Spotlight to parallelize queries. This can be done by running the above command multiple times with different names and ports. Alternatively you can use the existing docker compose file to start many instances automatically:
`sudo docker compose -f spotlight-compose.yml up -d`
This will create 16 DBPedia Spotlight instances. Comment out or delete parts of `spotlight-compose.yml` if you'd like fewer.

To stop instances run `sudo docker stop <name>`.

## Pre-training Datasets
We provide scripts for entity linking The Pile, ROOTS, C4, OpenWebText, and Wikipedia. To entity link OpenWebText or Wikipedia, use the following commands:

OpenWebText:
```console
python link_pretraining_data.py <output directory> \
                                --dataset openwebtext \
                                --nprocs <number of processes> \
                                --spotlight_ports <list of ports> \
                                --confidence 0.4 \
                                --support 20
```

Wikipedia:
```console
python link_pretraining_data.py <output directory> \
                                --dataset wikipedia \
                                --wikipedia_path <path to wikipedia HF Dataset> \
                                --nprocs <number of processes> \
                                --spotlight_ports <list of ports> \
                                --confidence 0.4 \
                                --support 20
```

This script will spin up `nprocs` processes and run examples from `dataset` through the the DBPedia Spotlight instances listening on `spotlight_ports` in parallel. We suggest specifying `--nprocs` to be 3x the number of DBPedia Spotlight instances you have running. Also note that the Wikipedia command assumes you've already constructed a HuggingFace Dataset locally.

To entity link The Pile, ROOTS, and C4, we suggest using `link_pretraining_data_chunked.py`. The examples in these datasets have much more variation in length and are generally much larger. This means that certain processes lag behind the other processes if they encounter long training examples, and these end up being a pretty severe bottleneck. To mitigate this, `link_pretraining_data_chunked.py` splits the pre-training dataset into smaller chunks and passes those to `link_pretraining_data.py`.
```console
python link_pretraining_data_chunked.py <output directory> \
                                        --dataset <the_pile | roots | c4> \
                                        --nprocs <number of processes> \
                                        --spotlight_ports <list of ports> \
                                        --confidence 0.4 \
                                        --support 20 \
                                        --chunk_size 1000000
```

Note: To run the script on ROOTS, you will need access to the dataset through BigScience and must specify your HuggingFace auth token with `--auth_token [token]`.

## Question Answering Datasets
To run DBPedia Spotlight on Trivia QA or Natural Questions use the following commands:

Trivia QA:
```console
python link_qa.py <output_directory> \
                  --dataset trivia_qa \
                  --dataset_subset <subset to entity link e.g., unfiltered, rc, etc.> \
                  --dataset_splits <list of splits to entity link e.g., train, validation, etc.> \
                  --nprocs <number of processes> \
                  --spotlight_ports <list of ports> \
                  --confidence 0.4 \
                  --support 20
```

Natural Questions:
```console
python link_qa.py <output_directory> \
                  --dataset nq_open \
                  --dataset_splits <list of splits to entity link e.g., train, validation, etc.> \
                  --nprocs <number of processes> \
                  --spotlight_ports <list of ports> \
                  --confidence 0.4 \
                  --support 20
```
