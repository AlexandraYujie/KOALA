import ujson
import logging



class Loader(object):
    @staticmethod
    def load_collection(collection_path):
        collection = []
        with open(collection_path, encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if line_idx % (1000) == 0:
                    print(f'{line_idx // 1000}K', end=' ', flush=True)
                pid, passage = line.strip('\n\r ').split('\t')
                assert pid == 'id' or int(pid) == line_idx, f"pid={pid}, line_idx={line_idx}"
                collection.append(passage)
        return collection

    @staticmethod
    def load_samples(triples_path):
        samples = []
        with open(triples_path) as f:
            for line in f:
                triple = ujson.loads(line)
                samples.append(triple)
        return samples

    @staticmethod
    def load_queries(queries_path):
        # queries = OrderedDict()
        logging.info(f">>> Loading the queries from {queries_path} ...")
        with open(queries_path, encoding="utf-8") as fp:
            query = fp.read().strip()
            assert query, "query is empty."
        return query

    @staticmethod
    def load_inference_collection(collection_path):
        text_list = []
        idx_list = []
        with open(collection_path, encoding="utf-8") as f:
            while True:
                line = f.readline().strip()
                if not line:
                    break
                idx, text = line.split('\t')
                idx = int(idx)
                assert (idx not in set(idx_list)), ("Query QID", idx, "is repeated!")
                text_list.append(text)
                idx_list.append(idx)
        logging.info(f">>> Got {len(text_list)} queries. All QIDs are unique.")
        assert len(text_list) == len(idx_list)
        return text_list, idx_list