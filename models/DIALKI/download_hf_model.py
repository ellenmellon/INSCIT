import os
import json
from dataclasses import dataclass, field
from transformers import AutoModel, AutoTokenizer, HfArgumentParser


@dataclass
class Arguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name: str = field(
        default="bert-base-uncased",
        metadata={"help": ""}
    )
    data_name: str = field(
        default="inscit",
        metadata={"help": ""}
    )

parser = HfArgumentParser((Arguments))
args = parser.parse_args_into_dataclasses()[0]
args.output_parent_dir = f"./{args.data_name}/pretrained_models"
output_dir = os.path.join(args.output_parent_dir, args.model_name)
if os.path.exists(output_dir):
    print(f'{output_dir} exists!')
    exit()

print('='*100)
print(' '*10, f'Download {args.model_name} model and tokenizer')
print('='*100)
m = AutoModel.from_pretrained(args.model_name)
t = AutoTokenizer.from_pretrained(args.model_name)

print('='*100)
print(' '*10, f'Save {args.model_name} model and tokenizer')
print('='*100)
m.save_pretrained(output_dir)
t.save_pretrained(output_dir)

config_file = os.path.join(output_dir, "config.json")
with open(config_file) as fin:
    cfg = json.loads(fin.read())
    cfg["return_dict"] = False
with open(config_file, "w") as fout:
    fout.write(json.dumps(cfg, indent=2))

