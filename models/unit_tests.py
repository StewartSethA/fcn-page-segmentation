# From https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765
def test_convnet():
  image = tf.placeholder(tf.float32, (None, 100, 100, 3))
  model = Model(image)
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  before = sess.run(tf.trainable_variables())
  _ = sess.run(model.train, feed_dict={
               image: np.ones((1, 100, 100, 3)),
               })
  after = sess.run(tf.trainable_variables())
  for b, a, n in zip(before, after):
      # Make sure something changed.
      assert (b != a).any()
      
def test_loss():
  in_tensor = tf.placeholder(tf.float32, (None, 3))
  labels = tf.placeholder(tf.int32, None, 1)
  model = Model(in_tensor, labels)
  sess = tf.Session()
  loss = sess.run(model.loss, feed_dict={
    in_tensor:np.ones(1, 3),
    labels:[[1]]
  })
  assert loss != 0
  
# Useful for models with intentionally non-trainable layers.
def test_gen_training():
  model = Model
  sess = tf.Session()
  gen_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='gen')
  des_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='des')
  before_gen = sess.run(gen_vars)
  before_des = sess.run(des_vars)
  # Train the generator.
  sess.run(model.train_gen)
  after_gen = sess.run(gen_vars)
  after_des = sess.run(des_vars)
  # Make sure the generator variables changed.
  for b,a in zip(before_gen, after_gen):
    assert (a != b).any()
  # Make sure descriminator did NOT change.
  for b,a in zip(before_des, after_des):
    assert (a == b).all()
