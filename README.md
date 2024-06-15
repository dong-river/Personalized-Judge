# Personalized-Judge

## Install environment
```bash
conda create --name judge python=3.9.16
conda activate judge
pip install -r requirements.txt
```

## Download Dataset
For Author Profiling (AP) on Public Reddit, please download synthetic_dataset.jsonl to the folder './ap' from this [link](https://github.com/eth-sri/llmprivacy/tree/main/data/synthetic). \
For Empathetic Conversation (EC), please download Track 3 and Track 4 data from [link](https://codalab.lisn.upsaclay.fr/competitions/18810#learn_the_details-datasets) and then artciles from [link](https://drive.google.com/file/d/1A-7XiLxqOiibZtyDzTkHejsCtnt55atZ/view). And put these 3 CSV under './ec'. \
For OpinionQA, please download the [data](https://worksheets.codalab.org/worksheets/0x6fb693719477478aac73fc07db333f69) to './opinions_qa'. 

## Run Experiments
```
python prism.py --num_sample 1000 --prompt_type with_persona --persona_features all_features --model gpt-4o
python opinionqa.py --num_sample 200 --prompt_type with_persona --model gpt-4o
python ap.py --model gpt-4o
python ec.py --num_article 300 --num_pair_per_article 5  --prompt_type all_features --model gpt-4o
```
--prompt_type controls the type of prompt to be used: with_persona, no_persona, and with_persona_with_tie\
--persona_features controls the number of personas to be used: all_features, key_features, least_imp_feature

More sample experiment can be found in exp/exp.sh

## Results and Visualization
All the results can be found under \outputs. The jupyter notebook to visualize the results can be found in visualization.ipynb.
