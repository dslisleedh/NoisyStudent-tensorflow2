import tensorflow as tf


class NoisyStudent(tf.keras.models.Model):
    def __init__(self,
                 num_classes
                 ):
        '''
        Original paper uses EffNet-L2 but i used EffNet-B4 for convenience
        No stochastic depth applied
        input_size : 224
        :param num_classes:
        '''
        super(NoisyStudent, self).__init__()
        self.num_classes = num_classes

        self.Aug = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.Resizing(256, 256),
            tf.keras.layers.experimental.preprocessing.RandomRotation(factor=.015),
            tf.keras.layers.experimental.preprocessing.RandomCrop(224, 224)
        ])
        self.Model1 = tf.keras.applications.efficientnet.EfficientNetB4(include_top=False)
        self.Model1 = tf.keras.Sequential([
            self.Model1,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(.2),
            tf.keras.layers.Dense(self.num_classes,
                                  activation='softmax'
                                  )
        ])
        self.Model1.build((None, 224, 224, 3))
        self.Model2 = tf.keras.applications.efficientnet.EfficientNetB4(include_top=False)
        self.Model2 = tf.keras.Sequential([
            self.Model2,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(.2),
            tf.keras.layers.Dense(self.num_classes,
                                  activation='softmax'
                                  )
        ])
        self.Model2.build((None, 224, 224, 3))
        self.initial = True
        self.Mode = 1

    def compile(self,
              optimizer,
              loss,
              metrics,
              **kwargs):
        super(NoisyStudent, self).compile()
        self.model1optimizer = optimizer
        self.model2optimizer = optimizer
        self.loss_fn = loss
        self.metrics_ = metrics

    @tf.function
    def train_step(self, data):
        labeled, x_unlabeled = data
        x_labeled, y_labeled = labeled
        if self.initial:
            # 1. Update teacher model
            with tf.GradientTape() as tape:
                y_pred = self.Model1(self.Aug(x_labeled,
                                              training=True
                                              ),
                                     training=True
                                     )
                loss = self.loss_fn(y_labeled, y_pred)
            grads = tape.gradient(loss, self.Model1.trainable_variables)
            self.model1optimizer.apply_gradients(
                zip(grads, self.Model1.trainable_variables)
            )
            # 2. Use normal teacher to generator label
            y_unlabeled = self.Model1(x_unlabeled,
                                      training=False
                                      )
            y_total = tf.concat([y_labeled, y_unlabeled],
                                axis=0
                                )
            # 3. learn student model generate learn from noisy data(unlabeled + labeled)
            with tf.GradientTape() as tape:
                y_pred = self.Model2(self.Aug(tf.concat([x_labeled, x_unlabeled],
                                                        axis = 0
                                                        ),
                                              training=True
                                              ),
                                     training=True
                                     )
                loss = self.loss_fn(y_total, y_pred)
            grads = tape.gradient(loss, self.Model2.trainable_variables)
            self.model2optimizer.apply_gradients(
                zip(grads, self.Model2.trainable_variables)
            )
            # 4. set student model to teacher and go back to step 2
            self.mode *= -1
            self.initial = False
        else:
            if self.mode == 1:
                # 2. Use normal teacher to generator label
                y_unlabeled = self.Model1(x_unlabeled,
                                          training=False
                                          )
                y_total = tf.concat([y_labeled, y_unlabeled],
                                    axis=0
                                    )
                # 3. learn student model generate learn from noisy data(unlabeled + labeled)
                with tf.GradientTape() as tape:
                    y_pred = self.Model2(self.Aug(tf.concat([x_labeled, x_unlabeled],
                                                            axis=0
                                                            ),
                                                  training=True
                                                  ),
                                         training=True
                                         )
                    loss = self.loss_fn(y_total, y_pred)
                grads = tape.gradient(loss, self.Model2.trainable_variables)
                self.model2optimizer.apply_gradients(
                    zip(grads, self.Model2.trainable_variables)
                )
                # 4. set student model to teacher and go back to step 2
                self.mode *= -1
            else:
                # 2. Use normal teacher to generator label
                y_unlabeled = self.Model2(x_unlabeled,
                                          training=False
                                          )
                y_total = tf.concat([y_labeled, y_unlabeled],
                                    axis=0
                                    )
                # 3. learn student model generate learn from noisy data(unlabeled + labeled)
                with tf.GradientTape() as tape:
                    y_pred = self.Model1(self.Aug(tf.concat([x_labeled, x_unlabeled],
                                                            axis=0
                                                            ),
                                                  training=True
                                                  ),
                                         training=True
                                         )
                    loss = self.loss_fn(y_total, y_pred)
                grads = tape.gradient(loss, self.Model1.trainable_variables)
                self.model1optimizer.apply_gradients(
                    zip(grads, self.Model1.trainable_variables)
                )
                # 4. set student model to teacher and go back to step 2
                self.mode *= -1

        return {'total_loss': loss,
                f'{self.metrics_.name}': self.metrics(y_total, y_pred)
                }

    @tf.function
    def test_step(self, data):
        x, y = data
        if self.mode == 1:
            y_pred = self.Model1(x,
                                 training=False
                                 )
            loss = self.loss_fn(y, y_pred)
            met = self.metrics_(y, y_pred)
        else:
            y_pred = self.Model2(x,
                                 training=False
                                 )
            loss = self.loss_fn(y, y_pred)
            met = self.metrics_(y, y_pred)

        return {'loss': loss,
                f'{self.metrics_.name}': met
                }

    @tf.function
    def call(self, inputs, training=None, mask=None):
        if self.Mode == 1:
            return self.Model1(inputs,
                               training=training
                               )
        else:
            return self.Model2(inputs,
                               training=training
                               )