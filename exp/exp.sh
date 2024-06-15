python prism.py --num_sample 1000 --prompt_type with_persona --persona_features all_features --model command-r-plus
python prism.py --num_sample 1000 --prompt_type with_persona --persona_features all_features --model gpt-3.5-turbo
python prism.py --num_sample 1000 --prompt_type with_persona --persona_features all_features --model gpt-4o

python prism.py --num_sample 1000 --prompt_type no_persona --persona_features all_features --model command-r-plus
python prism.py --num_sample 1000 --prompt_type with_persona_with_tie --persona_features all_features --model command-r-plus
python prism.py --num_sample 1000 --prompt_type no_confidence --persona_features all_features --model command-r-plus

python prism.py --num_sample 5000 --prompt_type with_persona --persona_features key_features --model command-r-plus
python prism.py --num_sample 5000 --prompt_type with_persona --persona_features least_imp_feature --model command-r-plus

python opinionqa.py --num_sample 200 --prompt_type with_persona --model command-r-plus
python opinionqa.py --num_sample 200 --prompt_type with_persona --model gpt-3.5-turbo
python opinionqa.py --num_sample 200 --prompt_type with_persona --model gpt-4o
python opinionqa.py --num_sample 200 --prompt_type with_persona --persona_features least_imp_feature --model gpt-4o
python opinionqa.py --num_sample 200 --prompt_type with_persona --persona_features key_features --model gpt-4o
python opinionqa.py --num_sample 200 --prompt_type with_persona --persona_features least_imp_feature --model command-r-plus
python opinionqa.py --num_sample 200 --prompt_type with_persona --persona_features key_features --model command-r-plus
python opinionqa.py --num_sample 200 --prompt_type no_persona --model command-r-plus
python opinionqa.py --num_sample 200 --prompt_type no_confidence --model command-r-plus

python ec.py --num_article 300 --num_pair_per_article 5  --prompt_type with_persona --model command-r-plus
python ec.py --num_article 300 --num_pair_per_article 5  --prompt_type with_persona --model gpt-3.5-turbo
python ec.py --num_article 300 --num_pair_per_article 5  --prompt_type with_persona --model gpt-4o

python ap.py --model gpt-3.5-turbo
python ap.py --model command-r-plus
python ap.py --model gpt-4o