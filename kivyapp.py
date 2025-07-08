from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window
import cv2
from ultralytics import YOLO
import time
from kivy.properties import StringProperty

class MainLayout(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.Isrunnning=False
        self.starttime = 0
        self.runningtime = 0
        self.finishtime = 0
        self.x2=0
        self.lastx2 = 0
        self.passing_border = 400
        
        self.orientation = 'vertical'

        self.num_box = BoxLayout(size_hint=(1, 0.1))
        self.nowtime = Label(text='0',font_size=40)
        self.num_box.add_widget(self.nowtime)
        self.add_widget(self.num_box)


        self.img_widget = Image(size_hint=(1, 0.8))
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
        self.ready = False

    def ready_detection(self, instance):
        if self.capture is None:
            self.capture = cv2.VideoCapture(0)
        if self.event is None:
            self.event = Clock.schedule_interval(self.update, 1.0 / 10)
            self.button_box.remove_widget(self.ready_btn)
            self.button_box.add_widget(self.start_btn,index=2)
            self.ready = True

    def passing_detection(self,instance):
        if not self.Isrunnning:
            if self.x2 >= self.passing_border and self.lastx2 < self.passing_border:
                self.start_detection('on_press')
    def start_detection(self,instance,):
        if not self.Isrunnning:
            self.Isrunnning=True#ここでtimeを取得してそこからの計算で時間を出す。より処理による誤差が減るはず
            self.starttime = round(time.time()*1000)
            self.ready = False

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
        #時間処理
        if not self.ready:
            self.nowtime.text = str(float((round(time.time()*1000)-self.starttime))/1000)
        else:
            self.nowtime.text = str(0)

        ret, frame = self.capture.read()
        frame = cv2.flip(frame,1)#インカメだと逆だったので調節
        if not ret:
            return
        # YOLO推論
        results = self.model(frame, show=False, classes=[0])
        # 検出結果の描画（例: 青枠）
        for result in results:
            for detection in result.boxes:
                x1,y1, self.x2, y2 = map(int, detection.xyxy[0])
                cv2.rectangle(frame, (x1,y1), (self.x2, y2), (255, 0, 0), 2)
        cv2.line(frame,(self.passing_border, 0), (self.passing_border, 600), (0, 0, 255), 5)
        self.passing_detection('on_press')#buttonは押された時の状態をinstanceのところに送っていた
        self.lastx2 = self.x2


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