import logging

import kivy
import kivy.app
import kivy.graphics as gfx
import kivy.uix.button as btn
import kivy.uix.gridlayout as grid
import kivy.uix.label as lbl
import kivy.uix.widget as wgt
import numpy

#import tensorflow as tf
#from PIL import Image, ImageChops

from sklearn.externals import joblib
from keras.models import load_model



class DigitApp(kivy.app.App):
  def __init__(self):
    super().__init__()
    
    #tf.logging.set_verbosity(tf.logging.WARN)
    self.load_dir = './data/digit-model'
    # self.neural_network = DigitNetwork().neural_network
    # saver = tf.train.Saver()
    # self.session = tf.Session()
    # checkpoint = tf.train.get_checkpoint_state(self.load_dir + '/')
    # if not checkpoint or not checkpoint.model_checkpoint_path:
    #   raise ImportError('Failed to restore previous TensorFlow checkpoint!')
    # saver.restore(self.session, checkpoint.model_checkpoint_path)
    self.pipeline = joblib.load("./pipeline.pkl")
    self.model = load_model("./model.h5")

    self.image_size = 28
    self.height = self.image_size * 10
    self.width = int(self.height * 1.5)
    kivy.config.Config.set('graphics', 'width', self.width)
    kivy.config.Config.set('graphics', 'height', self.height)
    self.painter=None
    self.guess_label=None

  def run(self):
    super().run()

  def stop(self):
    super().stop()
    self.session.close()

  def build(self):
    layout = grid.GridLayout(cols=2,
                             row_default_height=self.height, col_default_width=self.width * 2 // 3)
    self.painter = PaintWidget(width=self.height, height=self.height)
    clearbtn = btn.Button(text='Clear')
    clearbtn.bind(on_release=lambda _: self.clear_canvas())
    recognizebtn = btn.Button(text='Recognize')
    recognizebtn.bind(on_release=lambda _: self.predict())
    self.guess_label = lbl.Label(text='[color=ffffff]?[/color]', font_size='22px', markup=True)
    layout.add_widget(self.painter)
    panel = grid.GridLayout(rows=3, col_default_width=self.width // 3, row_default_height=self.height//3)
    panel.add_widget(self.guess_label)
    panel.add_widget(recognizebtn)
    panel.add_widget(clearbtn)
    layout.add_widget(panel)
    return layout

  def clear_canvas(self):
    self.painter.canvas.clear()

  def predict(self):
     images = self.images()
     predictions = self.model.predict_proba(images)
     self.display(predictions[0])

  def images(self):
    filepath = self.load_dir + '/drawing.png'
    self.painter.export_to_png(filepath)
    image = Image.open(filepath)
    image.load()
    background = Image.new(image.mode, image.size, (255, 255, 255, 255))
    diff = ImageChops.difference(image, background)
    mask = ImageChops.add(diff, diff, 2.0, -100)
    bbox = mask.getbbox()
    cropped = image.crop(bbox)
    cropped.thumbnail((int(self.image_size * 0.8), int(self.image_size * 0.8)), Image.ANTIALIAS)
    thumbnail = Image.new('RGBA', (self.image_size, self.image_size), (255, 255, 255, 255))
    thumbnail.paste(cropped,
                    ((self.image_size - cropped.size[0]) // 2,
                     (self.image_size - cropped.size[1]) // 2))
    thumbnail.save(self.load_dir + '/drawing_thumbnail.png')
    raw_image_data = numpy.asarray(thumbnail, dtype="int32")
    image_data = numpy.apply_along_axis(lambda rgba: [1 - sum(rgba[:3]) / (3 * 255)], 2, raw_image_data)
    image_data.reshape((self.image_size, self.image_size, 1))
    return [image_data]

  def display(self, prediction):
    guess = numpy.argmax(prediction)
    self.guess_label.text = '[color=ffffff]{}[/color]'.format(guess)
    logging.info('Prediction: {}'.format(guess))
    standardized_weights = (prediction - min(prediction)) / (max(prediction) - min(prediction))
    total_weight = sum(standardized_weights)
    probabilities = [weight / total_weight for weight in standardized_weights]
    logging.info('Confidence: {}%'.format(int(probabilities[guess] * 100)))


class PaintWidget(wgt.Widget):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    with self.canvas.before:
      gfx.Color(1, 1, 1)
      gfx.Rectangle(size=self.size, pos=self.pos)

  def on_touch_down(self, touch):
    with self.canvas:
      gfx.Color(0, 0, 0)
      touch.ud['line'] = gfx.Line(points=(touch.x, touch.y), width=5)

  def on_touch_move(self, touch):
    touch.ud['line'].points += [touch.x, touch.y]


DigitApp().run()
