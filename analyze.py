from koala.evaluation import *
from koala.config.config import DistillConfig, Config
from koala.config.config import AnalConfig
from koala.generate_score import generate_score
import argparse
from koala.utils.training import amend_config_by_args
from koala.utils.training import flatten_list_of_list

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--batch_size", type=int)
    args = parser.parse_args()
    return args



def get_texts_and_idxes():
    """
    Reads and processes the input data.

    This custom function loads textual data and returns it in a structured format,
    including a list of text entries and a corresponding list of indices.

    Returns:
        Tuple[List[str], List[int]]:
            - text_list: A list of text strings.
            - idx_list: A list of integer indices corresponding to each text entry.
    """

    text_list = ["二是积极推进科技创新。实施创新驱动战略，支持创新要素、创新资源向产业、园区、企业集聚，完善产业技术开发、产权交易保护、科技金融服务等平台，加快构建以市场为导向、企业为主体、产学研相结合的区域创新体系，推动科技成果商品化、资本化、产业化。重点实施“5个100科技计划”：鼓励企业自主创新，扶持发展100家创新能力突出、具有自主知识产权的创新型企业、知识产权示范企业；组建100个工程技术中心、重点实验室和产业技术创新战略联盟；实施100个重点科技攻关项目；实施100个重点技术改造项目；培养、引进100个高层次人才。启动建设中科院贵州现代资源技术研究成果转化中心。力争全社会研发经费占生产总值比重达1.8%，科技进步对经济增长贡献率达54.5%，高新技术产业增加值增长30%以上。",
                 "优化科技创新环境。制定科技投入、税收奖励、金融支持、创造和保护知识产权、科技创新基地与平台等政策规定，保障中长期科技发展规划的落实。自 主创新，人才为本。加快培养一批科技拔尖人才和研发团队。长期以来，科技人员在加快我市科技进步和经济社会发展中做出了突出贡献，赢得了全社会的赞誉。我 们要全力支持广大科技工作者在建设创新型城市的实践中建立新的业绩。",
                ]
    idx_list = [0,1,2]
    return text_list, idx_list




if __name__ == "__main__":
    args = parse_args()
    anal_config = AnalConfig()
    amend_config_by_args(anal_config, args)

    if "distill" in args.model_dir:
        model_config = DistillConfig()
    else:
        model_config = Config()
    configure_config(model_config, args.model_dir)
    amend_config_by_args(model_config, args, filter_keys=["model_dir"])  # using args to change the model_config
    text_list, idx_list = get_texts_and_idxes()
    score_list = generate_score(model_config, text_list, idx_list)
    for text, score in zip(text_list, score_list):
        print("score:", score, "\t", "text:", text)



