from dataclasses import dataclass

@dataclass
class AnalSetting():
    model_dir: str = ""
    @property
    def exp_name(self):
        split_root = self.model_dir.split("/")
        split_root = [i for i in split_root if i]
        exp_name = "_".join(split_root[2:])
        return exp_name


if __name__ == "__main__":
    pass

