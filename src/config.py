class Config:
    
    def __init__(self) -> None:
        self.preprocessing = {
            "step0": {
                "load_data_path": "tmp/data/WebQSP/data/WebQSP.train.json",
                "dump_data_path": "tmp/preprocessing/step0.json",
                "dump_data_folder": "tmp/preprocessing",
            },
            "step1": {
                "load_data_path": "tmp/preprocessing/step0.json",
                "dump_data_path": "tmp/preprocessing/step1.json"
            },
            "step2": {
                "load_data_path": "tmp/preprocessing/step1.json",
                "dump_data_path": "tmp/preprocessing/step2.json"
            },
            "step3": {
                "load_data_path": "tmp/preprocessing/step2.json",
                "dump_data_path": "tmp/retriever/train.csv",

                # "load_data_path": "tmp/preprocessing/step2.json",
                # "dump_data_path": "tmp/retriever/train.csv",

                # "sup_load_data_path": "tmp/preprocessing/supervised_data_train.json",
                # "sup_dump_data_path": "tmp/retriever/sup_train.csv",

                "unsup_load_data_path": "tmp/preprocessing/unsupervised_data.json",
                "unsup_dump_data_path": "tmp/retriever/unsup_train.csv",

                "dump_data_folder": "tmp/retriever",
            }
        }
        self.train_retriever = {
            "load_data_path": "tmp/retriever/train.csv",
            "dump_model_path": "tmp/model_ckpt/SimBERT",

            # "load_data_path": "tmp/retriever/train.csv",
            # "dump_model_path": "tmp/model_ckpt/SimBERT",

            # "sup_load_data_path": "tmp/retriever/sup_train.csv",
            # "sup_dump_model_path": "tmp/model_ckpt/sup_SimBERT",
            
            # "unsup_load_data_path": "tmp/retriever/unsup_train.csv",
            # "unsup_dump_model_path": "tmp/model_ckpt/unsup_SimBERT"
        }
        self.retriever_model_ckpt = self.train_retriever["dump_model_path"]
        self.retrieve_subgraph = {
            "load_data_folder": "tmp/data/origin_nsm_data/webqsp",
            "dump_data_folder": "tmp/reader_data/webqsp"
        }
        self.train_reader = {
            "load_data_path": "tmp/reader_data/webqsp/",
            "dump_model_path": "tmp/model_ckpt/nsm/",
        }
        self.retriever_finetune = {
            "checkpoint_dir": "tmp/model_ckpt/nsm/",
            "load_data_path": "tmp/reader_data/webqsp/",
            "dump_model_path": "tmp/model_ckpt/SimBERT/",
            "dump_data_path": "tmp/retriever/",
        }

cfg = Config()
