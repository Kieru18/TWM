def generate_model():
    from tensorflow.keras.regularizers import l2
    from keras import Input, Model
    from keras.layers import (Conv2D, BatchNormalization, Activation, MaxPooling2D,
                               Dropout, GlobalAveragePooling2D, Dense,
                               Multiply, Reshape)

    wd = 1e-4

    def conv_bn_relu(x, filters):
        x = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l2(wd))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    def se_block(x, filters, ratio=16):
        se = GlobalAveragePooling2D()(x)
        se = Reshape((1, 1, filters))(se)
        se = Dense(filters // ratio, activation='relu', use_bias=False)(se)
        se = Dense(filters, activation='sigmoid', use_bias=False)(se)
        return Multiply()([x, se])

    inputs = Input(shape=(32, 32, 3))

    # Block 1: 32x32 -> 16x16
    x = conv_bn_relu(inputs, 64)
    x = conv_bn_relu(x, 64)
    x = se_block(x, 64)
    x = MaxPooling2D(2, 2)(x)
    x = Dropout(0.2)(x)

    # Block 2: 16x16 -> 8x8
    x = conv_bn_relu(x, 128)
    x = conv_bn_relu(x, 128)
    x = se_block(x, 128)
    x = MaxPooling2D(2, 2)(x)
    x = Dropout(0.3)(x)

    # Block 3: 8x8 -> 4x4
    x = conv_bn_relu(x, 256)
    x = conv_bn_relu(x, 256)
    x = conv_bn_relu(x, 256)
    x = se_block(x, 256)
    x = MaxPooling2D(2, 2)(x)
    x = Dropout(0.4)(x)

    # Classifier head
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, kernel_regularizer=l2(wd))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(10, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.summary()

    total_steps = (40000 // 100) * 20
    warmup_steps = (40000 // 100) * 2

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.001,
        decay_steps=total_steps - warmup_steps,
        alpha=1e-6,
        warmup_steps=warmup_steps,
        warmup_target=0.001
    )

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=wd
    )
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        optimizer=optimizer,
        metrics=['accuracy']
    )

    return model