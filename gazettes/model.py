def build_model(configuration):
    """
    Build tensorflow model from a JSON file with the model configuration.
    """
    logging.info("Creating model...")

    encoder_input = tf.keras.layers.Input(shape=(max_text_length, embedding_dimensions))
    layer = None
    if rnn_type == "lstm":
        layer = tf.keras.layers.LSTM(
            units=dimensoes_espaco_latent, dropout=dropout, activation=activation
        )
    else:
        layer = tf.keras.layers.GRU(
            units=dimensoes_espaco_latent, dropout=dropout, activation=activation
        )
    encoder = None
    if bidirectional:
        encoder = tf.keras.layers.Bidirectional(layer, merge_mode="sum")(encoder_input)
    else:
        encoder = layer(encoder_input)

    decoder = tf.keras.layers.RepeatVector(max_text_length, name="repeater")(encoder)
    if rnn_type == "lstm":
        if bidirectional:
            decoder = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    units=embedding_dimensions,
                    return_sequences=True,
                    dropout=dropout,
                    activation=activation,
                ),
                merge_mode="sum",
            )(decoder)
        else:
            decoder = tf.keras.layers.LSTM(
                units=embedding_dimensions,
                return_sequences=True,
                dropout=dropout,
                activation=activation,
            )(decoder)
    else:
        if bidirectional:
            decoder = tf.keras.layers.Bidirectional(
                tf.keras.layers.GRU(
                    units=embedding_dimensions,
                    return_sequences=True,
                    dropout=dropout,
                    activation=activation,
                ),
                merge_mode="sum",
            )(decoder)
        else:
            decoder = tf.keras.layers.GRU(
                units=embedding_dimensions,
                return_sequences=True,
                dropout=dropout,
                activation=activation,
            )(decoder)

    model = tf.keras.Model(encoder_input, decoder, name=model_name)

    loss = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.RMSprop(learning_rate)
    metrics = [tf.keras.metrics.MeanSquaredError(), "accuracy"]

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model
