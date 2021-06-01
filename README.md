# CGRE

Codes and datasets for our paper [Distantly-Supervised Long-Tailed Relation Extraction Using Constraint Graphs](https://arxiv.org/abs/2105.11225)

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

#### train.json & test.json

    [
        {
            "text": "he is a son of vera and william lichtenberg of belle_harbor , queens .",
            "sub": {"id": "m.0ccvx", "name": "queens", "type": "GPE"},
            "obj": {"id": "m.05gf08", "name": "belle_harbor", "type": "GPE"},
            "rel": "/location/location/contains"
        },
        ...
    ]

#### rel2id.json

    {
        "NA": 0,
        "/location/neighborhood/neighborhood_of": 1,
        ...
    }

#### type2id.json

    {
        "NONE": 0,
        "CARDINAL": 1,
        ...
    }

#### constraint_graph.json

    {
        relation_1: [[head_type_1, tail_type_1], [head_type_2, tail_type_2], ...],
        ...
    }


## Training

    python train.py --name ${model_name}

The best AUC value normally appears when the pos_acc ranges from 0.6 to 0.8.

## Evaluation
    python eval.py --name ${model_name}

To reproduce our paper results, you can download our pre-trained model from [here](https://drive.google.com/file/d/1h-p8EBvvYbRpgwe7zTCH7oGfJQaSpneD/view?usp=sharing) and unzip it as the `pretrain` folder. Then you can directly evaluate it by the following command:

    python eval.py --name CGRE --save pretrain

## Results
The experimental results are logged in `./results`. The PR curves are stored in `./results/${model_name}.png`.

## Citation
Please cite our paper if you find it helpful.

    @article{liang2021distantly,
      title={Distantly-Supervised Long-Tailed Relation Extraction Using Constraint Graphs},
      author={Tianming Liang, Yang Liu, Xiaoyan Liu, Gaurav Sharma, Maozu Guo},
      journal={arXiv preprint arXiv:2105.11225},
      year={2021}
    }
