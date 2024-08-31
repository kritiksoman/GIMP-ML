import json
from PyQt6 import sip
from PyQt6.QtWidgets import *
from PyQt6.QtGui import QPixmap
import sys, os
import requests

# from tools.text_to_image import TextToImage
from PyQt6.QtCore import QProcess
from PyQt6 import QtCore
import subprocess
import threading


class Window(QMainWindow):
    progressSignal = QtCore.pyqtSignal(str)
    def __init__(self):
        super().__init__()

        self.p = None
       
        # set the title of main window
        self.setWindowTitle('GIMP-ML')

        # set the size of window
        self.Width = 800
        self.height = int(0.618 * self.Width)
        self.resize(self.Width, self.height)

        # add all widgets
        self.openai_button = QPushButton('OpenAI', self)
        # self.monodepth = QPushButton('Segment Anything', self)
        # self.btn_3 = QPushButton('Super resolution', self)
        # self.btn_4 = QPushButton('Monodepth', self)
        self.status = QLabel("Service Status: Starting.", self)
        

        self.openai_button.clicked.connect(self.text_to_image_action)
        # self.monodepth.clicked.connect(self.monodepth_action)
        # self.btn_3.clicked.connect(self.button3)
        # self.btn_4.clicked.connect(self.button4)

        # self.check
        self.check = list()
        self.models = list()

        # add tabs
        self.openai_tab = self.tab_1()
        self.tab2 = self.ui2()
        self.tab3 = self.ui3()
        self.tab4 = self.ui4()

        self.initUI()

        t = threading.Thread(target=self.fun)
        self.progressSignal.connect(self.status.setText)
        t.start()
        
    def fun(self):
        # def run():
        service_process = subprocess.Popen([os.path.join(os.path.dirname(__file__), ".venv", "Scripts", "python.exe"),#TODO: add .. when running in  VSCODE
                        os.path.join(os.path.dirname(__file__), "service.py")], 
                        stdout=subprocess.PIPE, universal_newlines=True, shell=True)
        # data = {
        #     "quality": model_version
        # }
        while True:
            try:
                response = requests.get(r"http://127.0.0.1:8000/status")
                response_data = response.json()
                if response_data['service'] == 'running':
                    self.progressSignal.emit("Service Status: Running.")
                    break
            except:
                pass


    def initUI(self):
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.openai_button)
        # left_layout.addWidget(self.monodepth)
        # left_layout.addWidget(self.btn_3)
        # left_layout.addWidget(self.btn_4)

        self.logo = QLabel()
        self.logo.setPixmap(QPixmap(os.path.join(os.path.dirname(__file__), "images", "plugin_logo.png")))
        left_layout.addWidget(self.logo)

        left_layout.addWidget(self.status)

        left_layout.addStretch(5)
        left_layout.setSpacing(20)
        left_widget = QWidget()
        left_widget.setLayout(left_layout)

        self.right_widget = QTabWidget()
        self.right_widget.tabBar().setObjectName("mainTab")

        self.right_widget.addTab(self.openai_tab, '')
        self.right_widget.addTab(self.tab2, '')
        self.right_widget.addTab(self.tab3, '')
        self.right_widget.addTab(self.tab4, '')

        self.right_widget.setCurrentIndex(0)
        self.right_widget.setStyleSheet('''QTabBar::tab{width: 0; \
            height: 0; margin: 0; padding: 0; border: none;}''')

        main_layout = QHBoxLayout()
        main_layout.addWidget(left_widget)
        main_layout.addWidget(self.right_widget)
        main_layout.setStretch(0, 40)
        main_layout.setStretch(1, 200)
        main_widget = QWidget()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    # ----------------- 
    # buttons

    def text_to_image_action(self):
        self.right_widget.setCurrentIndex(0)

    def monodepth_action(self):
        self.right_widget.setCurrentIndex(1)

    def button3(self):
        self.right_widget.setCurrentIndex(2)

    def button4(self):
        self.right_widget.setCurrentIndex(3)

    # ----------------- 
    # # pages
    # def download_models(self):
    #     for m, c in zip(self.models, self.check):
    #         if c.isChecked():
    #             print(c.text())
    #             # self.model = TextToImage(c.text())
    #             if self.p is None:  # No process running.
    #                 self.message("Executing process")
    #                 self.p = QProcess()  # Keep a reference to the QProcess (e.g. on self) while it's running.
    #                 self.p.readyReadStandardOutput.connect(self.handle_stdout)
    #                 self.p.readyReadStandardError.connect(self.handle_stderr)
    #                 self.p.stateChanged.connect(self.handle_state)
    #                 self.p.finished.connect(self.process_finished)  # Clean up once complete.
    #                 self.p.start("/Users/kritiksoman/gimp-test/.venv/bin/python3", ['/Users/kritiksoman/gimp-test/gimpml/tools/text_to_image.py', "CompVis/stable-diffusion-v1-4"])

    #             print("loaded")
    #     # pass

    # def handle_stderr(self):
    #     data = self.p.readAllStandardError()
    #     stderr = bytes(data).decode("utf8")
    #     self.message(stderr)

    # def handle_stdout(self):
    #     data = self.p.readAllStandardOutput()
    #     stdout = bytes(data).decode("utf8")
    #     self.message(stdout)

    # def handle_state(self, state):
    #     states = {
    #         QProcess.ProcessState.NotRunning: 'Not running',
    #         QProcess.ProcessState.Starting: 'Starting',
    #         QProcess.ProcessState.Running: 'Running',
    #     }
    #     state_name = states[state]
    #     self.message(f"State changed: {state_name}")

    # def process_finished(self):
    #     self.message("Process finished.")
    #     self.p = None
    
    def update_key(self):
        val = self.text.toPlainText()
        v = json.load(open(os.path.join(os.path.dirname(__file__), "..", "service", "config.json")))
        v['openai']['key'] = val
        with open(os.path.join(os.path.dirname(__file__), "..", "service", "config.json"), "w") as json_file:
            json.dump(v, json_file, indent=3)


    def message(self, s):
        self.text.appendPlainText(s)

    def tab_1(self):
        main_layout = QVBoxLayout()
        main_layout.addWidget(QLabel('Plugins: '))
        self.models = ["* Text to image", "* Extend image", "* Edit image with text"]
        for i in range(len(self.models)):
            main_layout.addWidget(QLabel(self.models[i]))
            # self.check.append(QCheckBox(self.models[i], self))
            # self.check1.toggled.connect(self.showDetails)
            # main_layout.addWidget(self.check[-1])

        verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        main_layout.addItem(verticalSpacer)

        main_layout.addWidget(QLabel("Configure key: "))
        self.text = QPlainTextEdit()
        
        try:
            with open(os.path.join(os.path.dirname(__file__), "config.json"), "r") as json_file:
                key = json.load(json_file)['openai']['key']
        except:
            key = ""

        self.text.appendPlainText(key)
        self.text.resize(400, 5)
        # self.text.setReadOnly(True)
        main_layout.addWidget(self.text)

        update_key_button = QPushButton('Update key', self)
        main_layout.addWidget(update_key_button)
        # name = self.text.toPlainText()
        update_key_button.clicked.connect(self.update_key)

        main_layout.addStretch(5)
        main = QWidget()
        main.setLayout(main_layout)
        return main

    def ui2(self):
        main_layout = QVBoxLayout()
        main_layout.addWidget(QLabel('page 2'))
        main_layout.addStretch(5)
        main = QWidget()
        main.setLayout(main_layout)
        return main

    def ui3(self):
        main_layout = QVBoxLayout()
        main_layout.addWidget(QLabel('page 3'))
        main_layout.addStretch(5)
        main = QWidget()
        main.setLayout(main_layout)
        return main

    def ui4(self):
        main_layout = QVBoxLayout()
        main_layout.addWidget(QLabel('page 4'))
        main_layout.addStretch(5)
        main = QWidget()
        main.setLayout(main_layout)
        return main


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Window()
    ex.show()
    sys.exit(app.exec())