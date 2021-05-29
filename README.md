# CGRE

Codes and datasets for our paper "Distantly-Supervised Long-Tailed Relation Extraction Using Constraint Graphs"

## Requirements

* Python 3.6+
* Pytorch 1.4.0+
* PyTorch Geometric (see https://github.com/rusty1s/pytorch_geometric for detail)

## Dataset

### Download

You can download the dataset from [here](https://drive.google.com/file/d/1TWiPmCbV6RcV-jhwbis7ljMnrnysqVF6/view?usp=sharing),
and unzip it as the `data` folder.

    tar -zxvf data.tar.gz

### Data Format

#### Training Data & Testing Data

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

#### Relation-ID Mapping Data   

    rel2id.json:

    {
        "NA": 0,
        "/location/neighborhood/neighborhood_of": 1,
        ...
    }

#### Type-ID Mapping Data   

    type2id.json:

    {
        "NONE": 0,
        "CARDINAL": 1,
        ...
    }

#### Constraint Data

    constraint_graph.json:

    {
        relation_1: [[head_type_1, tail_type_1], [head_type_2, tail_type_2], ...],
        ...
    }


## Training

    python train.py --name ${model_name}

The best AUC value normally appears when the pos_acc ranges from 0.6 to 0.8.

## Evaluation
    python eval.py --name ${model_name}

To reproduce our paper results, you can directly evaluate our pre-trained model by the following command:

    python eval.py --name CGRE --save pretrain

## Results
The experimental results are logged in `./log`. The PR curves are stored in `./log/${model_name}.png`.
