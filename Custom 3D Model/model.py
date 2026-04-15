import tensorflow as tf
from config import CFG

# class cusModel:
#     @staticmethod
#     def build(input_shape=(10, 224, 224, 3)):
#         model = tf.keras.Sequential([
#             tf.keras.Input(shape=input_shape),
#             tf.keras.layers.Conv3D(32, kernel_size=3, padding="same", activation="relu"),
#             tf.keras.layers.MaxPooling3D(),
#             tf.keras.layers.Conv3D(64, kernel_size=3, padding="same", activation="relu"),
#             tf.keras.layers.MaxPooling3D(),
#             tf.keras.layers.Conv3D(128, kernel_size=3, padding="same", activation="relu"),
#             tf.keras.layers.MaxPooling3D(),
#             tf.keras.layers.Dropout(0.3),
#             tf.keras.layers.GlobalAveragePooling3D(),
#             tf.keras.layers.Dense(1, activation="sigmoid")  
#             # tf.keras.layers.Dense(len(CFG.classes), activation="softmax")
#         ])

#         model.compile(
#             loss=tf.keras.losses.BinaryCrossentropy(),
#             # loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#             optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
#             metrics=["accuracy"]
#         )
#         return model

class cusModel:
    @staticmethod
    def build(input_shape=(10, 224, 224, 3)):
        inputs = tf.keras.Input(shape=input_shape)

        # -------- Block 1 --------
        x = tf.keras.layers.Conv3D(32, 3, padding="same")(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling3D((1, 2, 2))(x)

        # -------- Block 2 --------
        x = tf.keras.layers.Conv3D(64, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling3D((2, 2, 2))(x)

        # -------- Block 3 --------
        x = tf.keras.layers.Conv3D(128, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling3D((2, 2, 2))(x)

        x = tf.keras.layers.GlobalAveragePooling3D()(x)
        x = tf.keras.layers.Dropout(0.5)(x)

        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

        model = tf.keras.Model(inputs, outputs)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=["accuracy"]
        )
        return model


class PreModel:
    @staticmethod
    def build():
        if CFG.model_name == "vgg16":
            print("Using VGG16 model")
            net = tf.keras.applications.VGG16(
                include_top=False,
                weights="imagenet"
            )
        elif CFG.model_name == "vgg19":
            print("Using VGG19 model")
            net = tf.keras.applications.VGG19(
                include_top=False,
                weights="imagenet"
            )
        elif CFG.model_name == "resnet50":
            print("Using ResNet50 model")
            net = tf.keras.applications.ResNet50(
                include_top=False,
                weights="imagenet"
            )
        net.trainable = False

        model = tf.keras.Sequential([
            tf.keras.layers.Rescaling(255.0),
            tf.keras.layers.TimeDistributed(net),
            tf.keras.layers.Dense(1, activation='sigmoid'),
            tf.keras.layers.GlobalAveragePooling3D(), 
        ])

        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=['accuracy']
        )
        return model
