from transformers import (
    EncoderDecoderModel,
    AutoTokenizer,
    BertGenerationEncoder,
    BertGenerationDecoder,
    BertTokenizer,
)
from datasets import load_dataset

# dataset = load_dataset("pierreguillou/lener_br_finetuning_language_model")

# checkpoint = "bert-large-uncased"
# checkpoint = "neuralmind/bert-base-portuguese-cased"
checkpoint = "pierreguillou/bert-base-cased-pt-lenerbr"
# sentence =     "This is a long article to summarize"
sentence = "Isso é um longo artigo para ser sumarizado"

encoder = BertGenerationEncoder.from_pretrained(
    checkpoint  # , bos_token_id=101, eos_token_id=102
)
# add cross attention layers and use BERT's cls token as BOS token and sep token as EOS token
decoder = BertGenerationDecoder.from_pretrained(
    checkpoint,
    add_cross_attention=True,
    is_decoder=True,
    bos_token_id=101,
    eos_token_id=102,
)
bert2bert = EncoderDecoderModel(encoder=encoder, decoder=decoder)

# create tokenizer...
tokenizer = BertTokenizer.from_pretrained(checkpoint)
input_ids = tokenizer(sentence, add_special_tokens=False, return_tensors="pt").input_ids

outputs = bert2bert.generate(input_ids)
print(tokenizer.decode(outputs[0]))
