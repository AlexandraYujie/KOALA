# KOALA

## About KOALA

KOALA is a tool that leverages large language models (LLMs) to quantify texts in the social sciences. It adopts an LLM ranking + small model distillation method to enable text quantification in low-resource settings.

This repository accompanies the following paper:

📄 **Expert-level policy style measurement via knowledge distillation with large language model collaboration**   
📚 *Information Processing & Management, 2025*  
🔗 [https://doi.org/10.1016/j.ipm.2025.104090](https://doi.org/10.1016/j.ipm.2025.104090)


Currently, the KOALA repository supports:

* Near-zero annotation for automatic text quantification, reducing expert annotation time and cost
* Customizable text styles

## Quick Start

### Installation

```shell
pip install -r requirements.txt
```

### Prepare Raw Data

* KOALA only requires a `.jsonl` file with entries like `{"domain": "<domain>", "text": "<text>"}`. **No annotation** of the texts is necessary.

* **Note:** It’s acceptable to leave `<domain>` unspecified or assign all entries the same value, such as setting `<domain>` to the string `"whole"`.

* Example data (all sample data is in `data/raw/raw_data.jsonl`):

```json lines
{"domain": "agri", "text": "Agriculture is a strategic industry for national stability and people's well-being. Addressing the 'three rural issues' is crucial for overall socio-economic development. By promoting coordinated urban-rural development, using industry to support agriculture, and cities to support the countryside, we aim to boost farmers' incomes and enhance agricultural production capacity while ensuring food security. We are accelerating the transformation of traditional agriculture in irrigation areas into modern agriculture, and shifting the southern mountainous regions from traditional planting to a combination of modern ecological animal husbandry and farming."}
{"domain": "gov", "text": "According to the deployment of the Ninth Plenary Session of the Fifth Municipal Committee, the general requirement of government work this year is to thoroughly implement the spirit of the Fourth Plenary Session of the 17th CPC Central Committee and the Central Economic Work Conference, adhere to the scientific outlook on development, and focus on building an ecologically strong city and an industrially prosperous city. The goal is to enrich the people and strengthen the city, with stable and rapid economic development as the primary task. The focus is on transforming economic development modes and adjusting the economic structure, strengthening pillar industries, developing characteristic industries, fostering emerging industries, enhancing openness and innovation, advancing project construction and investment promotion, and comprehensively developing social undertakings to build a prosperous, open, civilized, and harmonious new Baishan."}
{"domain": "cul", "text": "Focus on developing the Taiji cultural industry. Make every effort to successfully host the Jiaozuo International Taijiquan Exchange Competition with innovation, influence, and effectiveness. Run the Taiji Culture Institute and expand enrollment to train a group of Taijiquan teachers for international Confucius Institutes. Based on offering Taijiquan courses in primary and secondary schools across the province, aim to expand nationwide. Plan a large-scale live performance show. Accelerate infrastructure construction in Chenjiagou, collect Taijiquan materials, and curate museum exhibits to improve service reception levels. Continue promoting the application for Taijiquan as a UNESCO Intangible Cultural Heritage. With continuous efforts, aim to build Jiaozuo into a world-famous 'Taiji holy land' and promote Taijiquan as a globally recognized cultural tourism brand."}
```

### Preprocess Raw Data

Preprocess and store data into:

* `data/processed/training/<domain>/collection.tsv`
* `data/processed/training/<domain>/grouped_data.jsonl` (unsorted)

```shell
# Ensure raw data is prepared in advance; default raw_data_path is data/raw/raw_data.json
cd path/to/koala
python preprocess.py
```

### Use LLM to Rank Texts and Split Train/Validation Sets

* Rank texts using an LLM and store results in `data/training/processed/<domain>/grouped_data.rank.jsonl`
* Split into `grouped_data.rank.train.jsonl` and `grouped_data.rank.valid.jsonl`

```shell
# In rank.py, you can configure language, style_name, style_definition, and LLM ranker parameters (api_key, base_url, train_ratio, etc.)
python rank.py
```

### Optional: Prepare Test Set and Expert Knowledge

**Test Set (optional):**

* The test set refers to **expert-annotated data** used to evaluate model performance. This is **not required**.

* If available, store it in: `data/processed/test/<domain>/collection.tsv` and `grouped_data.ranked.jsonl`.

* Note: Each line in `grouped_data.ranked.jsonl` should be pairwise data with **higher style first, lower style second**. See the sample under `data/processed/test/<domain>/`.

* If no annotated test set is available, **add `--no_test`** to your training command.

**Expert Knowledge (optional):**

* If available, edit `data/queries/0.txt` with language-style-related expert knowledge.
* If not available, **add `--random_query`** to your training command (you don’t need to delete or modify `0.txt`).

### Model Training

* If only one domain is used, we recommend saving the single-domain model and skipping second-stage distillation.

#### First-Stage Distillation

* Trained domain-specific models are saved to:
  `experiment/model/<domain>/<method>/<base_model_name>/<datetime>/`
* After training, the domain model performs inference on `collection.tsv`, and results are stored as:
  `experiment/model/<domain>/<method>/<base_model_name>/<datetime>/idx_value_result_<domain>.json`
  This will be used for second-stage distillation.

**Parameter Descriptions:**

* `domain`: corresponds to `<domain>` in `data/raw/raw_data.jsonl`
* `method`: defaults to `sbert` (sentence-BERT)
* `base_model_name`: name of model in `models/`, e.g., Huggingface’s `sbert-base-chinese-nli` should be in `models/sbert-base-chinese-nli/`
* `inference (bool)`: performs inference on `collection.tsv`, required for second-stage distillation

**Optional Arguments:**

* `--no_model_saving`: do not save model
* `--random_query`: no expert knowledge provided
* `--no_test`: no expert-labeled test set provided

```shell
# First-stage distillation
python -u training.py --domain edu --method sbert --base_model_name sbert-base-chinese-nli --batch_size 16 --epoch 3 --inference
python -u training.py --domain tech --method sbert --base_model_name sbert-base-chinese-nli --batch_size 16 --epoch 3 --inference
# ...

# Optional flags:
# --no_model_saving
# --random_query
# --no_test
```

#### Second-Stage Distillation

* This stage distills the first-stage domain models.
* Final model is saved at:
  `experiment/model/distill/<datetime>/`

**Preparation:**

1. Modify `experiment/exp_summary/name_idx_value_roots_dic.json` by updating the value of `"sbert.sbert-base-chinese-nli"` to point to the result file from first-stage distillation, such as:
   `experiment/model/<domain>/<method>/<base_model_name>/<datetime>/idx_value_result_<domain>.json`

2. In `training.py`, update:

   * `mixed_domains` (domains for optional testing)
   * `domain_seq` (domains for distillation training)

```shell
# Make sure name_idx_value_roots_dic.json is updated
# Start second-stage training:
python -u training.py --method sbert --base_model_name sbert-base-chinese-nli --batch_size 32 --epoch 1 --distill

# Optional flags:
# --no_model_saving
# --random_query
# --no_test
```

## Model Inference

**Parameter Note:**

* `model_dir` can point to either a first-stage domain model or a second-stage distillation model.

```shell
python -u analyze.py --batch_size 32 --model_dir experiment/model/distill/<datetime>/
```

## 📄 Citation

If you use this work in your research, please cite our paper:

```bibtex
@article{zhang2025expert,
  title={Expert-level policy style measurement via knowledge distillation with large language model collaboration},
  author={Zhang, Yujie and Huang, Biao and Yuan, Weikang and Jiang, Zhuoren and Peng, Longsheng and Chen, Shuai and Tan-Soo, Jie-Sheng},
  journal={Information Processing \& Management},
  volume={62},
  number={4},
  pages={104090},
  year={2025},
  publisher={Elsevier}
}
```


## Contact

For questions or collaborations, please contact: [jiangzhuoren@zju.edu.cn](mailto:jiangzhuoren@zju.edu.cn)
