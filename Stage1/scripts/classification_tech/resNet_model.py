def get_model():
    pass_layer = keras.layers.Lambda(lambda x:x)

    class RULayer(keras.layers.Layer):
        def __init__(self, fliters, kernel_shape, strides = 1, activation = "relu", **kwargs):
            super().__init__(**kwargs)
            self.activation = keras.activations.get(activation)
            self.non_skip_layers = [
                keras.layers.Conv2D(filters, kernal_shape, strides = strides, padding = "same", use_bias = False),
                keras.layers.BatchNormalization(),
                self.activation,
                keras.layers.Conv2D(filters, kernel_shape, strides = 1, padding = "same", use_bias = False),
                keras.layers.BatchNormalization()
            ]

            if strides > 1:
                self.skip_layers = keras.layers.Conv2D(filters, 1, strides = strides, padding = "same", use_bias = False)
            
            else:
                self.skip_layers = pass_layer
        
        def call(self, inputs):
            non_skip_output = inputs
            for layer in self.non_skip_layers:
                non_skip_output = layer(non_skip_output)
            
            skip_output = self.skip_layer(inputs)

            output = non_skip_output + skip_output
            return self.activation(output)

        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(32, (5, 10), strides = 2, input_shape = input_shape, padding = "same", use_bias = False))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation("relu"))
        model.add(keras.layers.AveragePooling2D(pool_size = (2,4), strides = 2, padding = "same"))

        prev_filters = 32
        for num_filters in [32, 32, 32, 64, 64, 64, 64, 128, 128, 128, 128, 128, 128, 256, 256, 256]:
            if (num_filters == prev_filters):
                strides = 1
            else:
                strides = 2
            model.add(RULayer(num_filters, (2, 5), strides))
            prev_filters = num_filters
            
        model.add(keras.layers.GlobalAvgPool2D())
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(10, activation = "softmax"))
        
    return model
