# CGRE

Codes and datasets for our paper "Distantly-Supervised Long-Tailed Relation Extraction Using Constraint Graphs"

## Requirements

* Python 3.6+
* Pytorch 1.4.0+
* PyTorch Geometric (see https://github.com/rusty1s/pytorch_geometric for detail)

## Dataset

Download the datasets and pretrained word embeddings from [here](https://github.com/thunlp/HNRE/tree/master/raw_data), and extract them under `data` folder.

## Training & Evaluation

Vanilla CGRE consists of PCNN and GCN, but we also provide some different backbone models: CNN, PCNN and Bert for sentence encoding and GCN, GAT and SAGE for graph encoding (see configuration files in `config/` for details). For example, you can try CNN+GAT on NYT-520K by the following command:

    python train.py --config 520K_CNN_GAT.yaml
and
    python eval.py --config 520K_CNN_GAT.yaml

## Results
PR curves are stored in `Curves/`.

## Data Format

### Training Data & Testing Data

    train.json & test.json: 

    [
        {
            "text": "he is a son of vera and william lichtenberg of belle_harbor , queens .",
            "sub": {"id": "m.0ccvx", "name": "queens", "type": "GPE"},
            "obj": {"id": "m.05gf08", "name": "belle_harbor", "type": "GPE"},
            "rel": "/location/location/contains"
        },
        ...
    ]

### Relation-ID Mapping Data   

    rel2id.json:

    {
        "NA": 0,
        "/location/neighborhood/neighborhood_of": 1,
        ...
    }

### Type-ID Mapping Data   

    type2id.json:

    {
        "NONE": 0,
        "CARDINAL": 1,
        ...
    }

### Constraint Data

    constraint_graph.json:

    {
        relation_1: [[head_type_1, tail_type_1], [head_type_2, tail_type_2], ...],
        ...
    }
