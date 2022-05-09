from transformers import AutoTokenizer


def load_bert_tokenizer(model_checkpoint: str, vocab_file: str):
    return AutoTokenizer.from_pretrained(
        model_checkpoint,
        use_fast=False,
        vocab_file=vocab_file,
        clean_text=True,
        do_lower_case=True,
    )


class TextBertAutoencoderWikipediaDataset(tf.data.Dataset):
    def __new__(
        cls,
        data_dir: str,
        batch_size: int = 1000,
        max_text_length: int = 64,
        vocabulary: str = None,
        vocabulary_size: int = 0,
        num_parallel_calls: int = tf.data.AUTOTUNE,
        model_checkpoint: str = "neuralmind/bert-base-portuguese-cased",
    ):
        dataset = WikipediaDataset(data_dir)
        tokenizer = load_bert_tokenizer(model_checkpoint, vocabulary)

        def preprocess_text(text):
            tokenizer_output = tokenizer(
                text.numpy().decode("utf8"),
                padding="max_length",
                truncation=True,
                max_length=max_text_length,
            )
            return (
                tokenizer_output["input_ids"],
                tokenizer_output["token_type_ids"],
                tokenizer_output["attention_mask"],
                tokenizer_output["input_ids"],
            )

        def tf_python_preprocess_text(text):
            preprocessed_text = tf.py_function(
                preprocess_text,
                [text],
                [
                    tf.TensorSpec(
                        shape=(max_text_length,), dtype=tf.int32, name="input_ids"
                    ),
                    tf.TensorSpec(
                        shape=(max_text_length,),
                        dtype=tf.int32,
                        name="token_type_ids",
                    ),
                    tf.TensorSpec(
                        shape=(max_text_length,),
                        dtype=tf.int32,
                        name="attention_mask",
                    ),
                    tf.TensorSpec(
                        shape=(max_text_length,), dtype=tf.int32, name="target"
                    ),
                ],
            )
            return [
                tf.reshape(
                    tensor,
                    [
                        max_text_length,
                    ],
                )
                for tensor in preprocessed_text
            ]

        def tf_preprocess_text(batch):
            return tf.map_fn(
                fn=tf_python_preprocess_text,
                elems=batch,
                fn_output_signature=[
                    tf.TensorSpec(
                        shape=(max_text_length,), dtype=tf.int32, name="input_ids"
                    ),
                    tf.TensorSpec(
                        shape=(max_text_length,),
                        dtype=tf.int32,
                        name="token_type_ids",
                    ),
                    tf.TensorSpec(
                        shape=(max_text_length,),
                        dtype=tf.int32,
                        name="attention_mask",
                    ),
                    tf.TensorSpec(
                        shape=(max_text_length,), dtype=tf.int32, name="target"
                    ),
                ],
            )

        dataset = dataset.map(
            tf_preprocess_text,
            num_parallel_calls=num_parallel_calls,
            deterministic=False,
        )
        if has_cache_enable():
            dataset.cache(
                get_cache_dir(data_dir, "transformer_preprocessing"),
            )

        def organize_targets(input_ids, token_type_ids, attention_mask, target):
            return (
                (input_ids, token_type_ids, attention_mask),
                target,
                # tf.one_hot(target, vocabulary_size),
            )

        def onehot_target(inputs, target):
            return (
                inputs,
                tf.one_hot(target, vocabulary_size),
            )

        dataset = dataset.map(organize_targets)
        logging.info(dataset.element_spec)
        dataset = dataset.map(onehot_target)
        if has_cache_enable():
            dataset.cache(get_cache_dir(data_dir, "transformer_one_hot_target"))
        return dataset
