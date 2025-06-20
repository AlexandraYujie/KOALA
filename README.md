# KOALA

## 了解KOALA
koala是利用大模型实现社科文本量化的工具，它通过使用大模型排序+小模型蒸馏的方法，实现低资源场景下的文本量化。
our paper:


目前KOALA仓库可以实现：
* 几近0标注自动实现文本量化，减轻专家标注时间和成本
* 文本风格自定义

## 快速启动

### Installation
```shell
pip install -r requirements.txt
```

### 准备raw data

* koala仅需要构建诸如`{"domain": "<domain>", "text": "<text>"}`的jsonl文件，可以不用进行**任何的**文本标注。

* **注：** 不指定具体的`<domain>`或仅指定单一的`<domain>`是可接受的，如`<domain>`均设置为字符串`"whole"`

* 数据示例（全部的数据样例在`data/raw/raw_data.jsonl`中展示）：
```json lines
{"domain": "agri", "text": "农业是安天下、稳民心的战略产业。解决好“三农”问题，事关经济社会发展全局。坚持城乡统筹，以工促农、以城带乡，突出农民持续增收和提高农业综合生产能力、确保粮食安全两大主题，加快实现引黄灌区由传统农业向现代农业、南部山区由传统种植业向现代生态畜牧业和种植业相结合转变。"}
{"domain": "gov", "text": "按照市委五届九次全会的部署，今年政府工作的总体要求是：深入贯彻落实党的十七届四中全会和中央经济工作会议精神，努力践行科学发展观，坚持生态立市、工业强市，以富民强市为目标，以保持经济平稳快速发展为首要任务，以推动经济发展方式转变和经济结构调整为主攻方向，做大做强支柱产业，发展壮大特色产业，大力培育新兴产业，不断提升对外开放水平和自主创新能力，全力推进项目建设、招商引资，全面发展社会各项事业，努力建设富裕开放、文明和谐的新白山。"}
{"domain": "cul", "text": "突出抓好太极文化产业开发。全力以赴办好焦作国际太极拳交流大赛，确保有新意、有影响、有效果。办好太极文化学院，扩大招生规模，为国际孔子学院培养一批太极拳师。在全省中小学开设太极拳课程基础上，争取向全国推广。研究策划大型实景演艺节目。加快陈家沟基础设施建设，做好太极拳资料搜集和博物馆布展工作，提高陈家沟接待服务水平。继续推进太极拳申报世界非物质文化遗产工作。通过几年努力，将焦作市打造成世界知名的“太极圣地”，将太极拳打造成为具有国际知名度和影响力的文化旅游品牌，让太极拳走向世界。"}
```

### 处理raw data

将数据**预处理**并存储至`data/processed/training/<domain>/`下的`collection.tsv`和`grouped_data.jsonl`（未经过排序）中。

```shell
# 请预先准备好raw data，默认的raw_data_path为data/raw/raw_data.json
cd koala所在目录
python preprocess.py
```

### 使用大模型进行数据文本排序，并分割训练集和验证集
* 将数据用**大模型进行排序**，并存储至`data/training/processed/<domain>/grouped_data.rank.jsonl`中
* 将数据分割成`grouped_data.rank.train.jsonl`和`grouped_data.rank.valid.jsonl`
```shell
# 在rank.py文件中，可以设置language，style_name，style_definition以及llm ranker相关参数（api_key, base_url, train_ratio）等
python rank.py
```

### 非必选项：准备测试集（optional）和专家知识（optional）

测试集（optional）：
* 这里的测试集是指用**有专家标注的数据**来查看模型的效果，此项**非**必选项
* 如有，请存储至`data/processed/test/<domain>/`路径下`collection.tsv`和`grouped_data.ranked.jsonl`文件下；注意`grouped_data.ranked.jsonl`的每一行数据构造是高文本风格在前，低文本风格在后的pairwise data，详见`data/processed/test/<domain>/`路径下的数据示例。
* 如没有标注测试集，请注意添加`--no_test`至模型训练的command line中

专家知识（optional）：
* 如有，请在`data/queries/0.txt`下对文本进行修改，改成语言风格相关的专家知识
* 如没有专家知识，请注意添加`--random_query`至模型训练的command line中，不用删除`data/queries/0.txt`文件及其内容。

### 模型训练
* 注意若仅有一个domain，建议保存一阶段领域模型，不需要使用二阶段继续蒸馏

#### 一阶段蒸馏
* 训练得到的模型是领域模型，被存储至`experiment/model/<domain>/<method>/<base_model_name>/<datetime>/`中
* 一阶段领域模型训练完成后，将读取`data/processed/training/<domain>/`下的`collection.tsv`文件，使用领域模型进行逐条推理，这将在二阶段蒸馏将被使用。领域模型的推理结果**存储位置**是：`experiment/model/<domain>/<method>/<base_model_name>/<datetime>/idx_value_result_<domain>.json`


参数说明：
* domain：与`data/raw/raw_data.jsonl`中的`<domain>`相对应
* method：默认是sbert，即sentence-BERT
* base_model_name：加载`models/`文件夹下的model，huggingface model`sbert-base-chinese-nli`存储在`models/sbert-base-chinese-nli/`文件夹下
* inference (bool)：读取`data/processed/training/<domain>/`下的`collection.tsv`文件进行逐条推理，这在二阶段distill将被使用

* **no_model_saving (bool)**：如**不存储模型**，可添加参数 **--no_model_saving**
* **random_query (bool)**：如无**专家知识**，可添加参数 **--random_query**
* **no_test (bool)**：如无**专家标注测试集**，可添加参数 **--no_test**
```shell
# 一阶段distill
python -u training.py --domain edu --method sbert --base_model_name sbert-base-chinese-nli --batch_size 16 --epoch 3 --inference
python -u training.py --domain tech --method sbert --base_model_name sbert-base-chinese-nli --batch_size 16 --epoch 3 --inference
# ...

# 如无需存储领域模型，请添加--no_model_saving
# 如无专家知识，请添加参数--random_query
# 如无专家标注测试集，可添加参数--no_test
```
#### 二阶段蒸馏
* 该阶段将蒸馏一阶段领域模型
* 模型将存储至：`experiment/model/distill/<datetime>/`

准备工作：

1. 在`experiment/exp_summary/name_idx_value_roots_dic.json`中修改`"sbert.sbert-base-chinese-nli"`的value，填充一阶段蒸馏结果文件名。文件名诸如：`experiment/model/<domain>/<method>/<base_model_name>/<datetime>/idx_value_result_<domain>.json`
2. 在`training.py`文件下修改`mixed_domains`（domains used for optional testing）和`domain_seq`（domains used for distillation training）参数

```shell
# 请先对experiment/exp_summary/name_idx_value_roots_dic.json进行修改
# 模型训练：
python -u training.py --method sbert --base_model_name sbert-base-chinese-nli --batch_size 32 --epoch 1 --distill

# 如无需存储领域模型，请添加--no_model_saving
# 如无专家知识，请添加参数--random_query
# 如无专家标注测试集，可添加参数--no_test
```

## 模型推理
参数说明：
* 这里model_dir可以填写一阶段领域模型路径，也可以填写二阶段蒸馏模型路径。
```shell
python -u analyze.py --batch_size 32 --model_dir experiment/model/distill/<datetime>/
```
# test
