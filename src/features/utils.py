import numpy as np
import tensorflow as tf

class DataGenerator():
  def __init__(self, input_width, label_width, offset, label_columns=None, df=None):

    if df is None:
      raise TypeError("No DataFrame provided.")

    # Store the raw data.
    n = len(df)
    self.train_df = df[0:int(n*0.7)]
    self.val_df = df[int(n*0.7):int(n*0.9)]
    self.test_df = df[int(n*0.9):]

    # window sliding parameters
    self.input_width = input_width
    self.label_width = label_width
    self.offset = offset

    self.total_width = self.input_width + self.offset
    self.input_slice = slice(0,input_width)
    self.input_indices = np.arange(self.total_width)[self.input_slice] # slide from beginning to the length of the input
    
    self.label_start = self.total_width - self.label_width
    self.label_slice = slice(self.label_start,None)
    self.label_indices = np.arange(self.total_width)[self.label_slice] # slide from the end from the last nth length of label

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(self.train_df.columns)}

  def __repr__(self):
    return '\n'.join(
        [f'Total Sequence Width:{self.total_width}', 
         f'Input slices:{self.input_indices}', 
         f'Label slices:{self.label_indices}',
         f'Label column name(s): {self.label_columns}'])

  def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.label_slice, :]
    if self.label_columns is not None:
      labels = tf.stack(
          [labels[:, :, self.column_indices[name]] for name in self.label_columns],
          axis=-1)

    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels
  
  def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_width,
        sequence_stride=1,
        shuffle=True,
        batch_size=32,)

    ds = ds.map(self.split_window)

    return ds

  @property
  def train(self):
    return self.make_dataset(self.train_df)

  @property
  def val(self):
    return self.make_dataset(self.val_df)

  @property
  def test(self):
    return self.make_dataset(self.test_df)

  @property
  def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
      # No example batch was found, so get one from the `.train` dataset
      result = next(iter(self.train))
      # And cache it for next time
      self._example = result
    return result
