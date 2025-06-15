from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window
import cv2
from ultralytics import YOLO

class MainLayout(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.img_widget = Image(size_hint=(1, 0.9))
        self.add_widget(self.img_widget)
        self.button_box = BoxLayout(size_hint=(1, 0.1))
        self.start_btn = Button(text='Start')
        self.ready_btn = Button(text='Ready')
        self.stop_btn = Button(text='Stop')
        self.lap_btn = Button(text='Lap')
        self.button_box.add_widget(self.ready_btn)
        self.button_box.add_widget(self.stop_btn)
        self.button_box.add_widget(self.lap_btn)
        self.add_widget(self.button_box)

        self.ready_btn.bind(on_press=self.ready_detection)
        self.start_btn.bind(on_press=self.start_detection)
        self.stop_btn.bind(on_press=self.stop_detection)

        self.model = YOLO("yolov8n.pt")
        self.model.to("cuda")
        self.capture = None
        self.event = None

        self.Isrunnning=False

    def ready_detection(self, instance):
        if self.capture is None:
            self.capture = cv2.VideoCapture(0)
        if self.event is None:
            self.event = Clock.schedule_interval(self.update, 1.0 / 10)
            self.button_box.remove_widget(self.ready_btn)
            self.button_box.add_widget(self.start_btn,index=2)

    def start_detection(self,instance):
        self.Isrunnning=True

    def stop_detection(self, instance):
        if self.event:
            self.event.cancel()
            self.event = None
            self.button_box.remove_widget(self.start_btn)
            self.button_box.add_widget(self.ready_btn,index=2)
        if self.capture:
            self.capture.release()
            self.capture = None
        self.Isrunnning=False

    def update(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            return
        # YOLO推論
        results = self.model(frame, show=False, classes=[0])
        # 検出結果の描画（例: 青枠）
        for result in results:
            for detection in result.boxes:
                x1, y1, x2, y2 = map(int, detection.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        #Opencvでリサイズ
        width, height = Window.size
        resized_frame = cv2.resize(frame, (width, height))
        # Kivy用に変換
        buf = cv2.flip(resized_frame, 0)
        buf = cv2.cvtColor(buf, cv2.COLOR_BGR2RGB)
        texture = Texture.create(size=(buf.shape[1], buf.shape[0]), colorfmt='rgb')
        texture.blit_buffer(buf.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        self.img_widget.texture = texture

class MainApp(App):
    def build(self):
        return MainLayout()

if __name__ == '__main__':
    MainApp().run()