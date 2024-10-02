import json
from PyQt6 import sip
from PyQt6.QtWidgets import *
from PyQt6.QtGui import QPixmap
import sys, os
import requests
import ctypes
import webbrowser
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtGui import QAction
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import QUrl
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCloseEvent
from PyQt6 import QtCore
import subprocess
import threading
import os
from Foundation import NSBundle


# Hide the application from the macOS dock
bundle = NSBundle.mainBundle()
info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
info['LSUIElement'] = '1'

# # ctypes.windll.shcore.SetProcessDpiAwareness(1)
# myappid = u'mycompany.myproduct.subproduct.version' # arbitrary string
# # ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
# ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('company.app.1')
    
class TrayWindow(QMainWindow):
    progressSignal = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()

        # set the title of main window
        self.setWindowTitle('GIMP-ML')
        # Disable minimize button
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowTitleHint | Qt.WindowType.CustomizeWindowHint | Qt.WindowType.WindowCloseButtonHint | Qt.WindowType.WindowMaximizeButtonHint)
        self.setWindowIcon(QIcon(os.path.join(os.path.dirname(__file__), "images", "icon.png")))
        self.service_process = None

        # set the size of window
        self.Width = 800
        self.height = int(0.618 * self.Width)
        self.resize(self.Width, self.height)

        # add all widgets for the left and the right panel
        self.openai_button = QPushButton('OpenAI', self)
        self.status = QLabel("Service Status: Starting.", self)
        

        self.openai_button.clicked.connect(self.activate_openai_panel)
        # self.monodepth.clicked.connect(self.monodepth_action)

        self.models = list()

        # add tabs
        self.openai_tab = self.tab_1()
        self.tab2 = self.ui2()

        self.initUI()


        # Set the window flag to hide from dock
        # self.setWindowFlags(Qt.WindowType.Tool)

        # System Tray Minimize
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(QIcon(os.path.join(os.path.dirname(__file__), "images", "icon.png")))
        tray_menu = QMenu() # Create the tray menu
        restore_action = QAction("Show window", self)
        restore_action.triggered.connect(self.show_raise_app)
        tray_menu.addAction(restore_action)

        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.quit_app)#QApplication.instance().quit)
        tray_menu.addAction(quit_action)

        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.setVisible(True)

        # GIMP-ML fast api service
        service_thread = threading.Thread(target=self.test_start_service)
        self.progressSignal.connect(self.status.setText)
        service_thread.start()
        
    def closeEvent(self, event):
        # Runs when window is closed
        event.ignore()
        self.hide()
        self.tray_icon.showMessage(
            "Minimize to Tray",
            "GIMP-ML was minimized to tray",
            QSystemTrayIcon.MessageIcon.Information,
            2000
        )

    # def closeEvent(self, event: QCloseEvent):
    #     # Terminate the subprocess
    #     self.service_process.terminate()
    #     self.service_process.wait()
    #     event.accept()

    def show_raise_app(self):
        self.show()
        self.raise_()
        self.activateWindow()        

    def quit_app(self):
        self.service_process.terminate()
        self.service_process.wait()
        QApplication.instance().quit()

    def test_start_service(self):
        try:
            response = requests.get(r"http://127.0.0.1:8000/status")
            response_data = response.json()
            if response_data['service'] == 'running':
                self.progressSignal.emit("Service Status: Running.")
        except:
            if sys.platform == "linux" or sys.platform == "linux2":
                self.service_process = subprocess.Popen([os.path.join(os.path.dirname(__file__), "..", "..", ".venv", "bin", "python"), #TODO: add .. when running in  VSCODE
                os.path.join(os.path.dirname(__file__), "service.py")], 
                stdout=subprocess.PIPE, universal_newlines=True, shell=True)
            elif sys.platform == "darwin":
                self.service_process = subprocess.Popen([os.path.join(os.path.dirname(__file__), "..", "..", ".venv", "bin", "python"), #TODO: add .. when running in  VSCODE
                os.path.join(os.path.dirname(__file__), "service.py")], 
                stdout=subprocess.PIPE, universal_newlines=True, shell=False)
            elif sys.platform == "win32":
                self.service_process = subprocess.Popen([os.path.join(os.path.dirname(__file__), "..", "..", ".venv", "Scripts", "python.exe"),#TODO: add .. when running in  VSCODE
                            os.path.join(os.path.dirname(__file__), "service.py")], 
                            stdout=subprocess.PIPE, universal_newlines=True, shell=True)
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

    def activate_openai_panel(self):
        self.right_widget.setCurrentIndex(0)

    def monodepth_action(self):
        self.right_widget.setCurrentIndex(1)

    def button3(self):
        self.right_widget.setCurrentIndex(2)

    def button4(self):
        self.right_widget.setCurrentIndex(3)

    def message(self, s):
        self.openai_key.appendPlainText(s)

    def tab_1(self):
        main = QWidget()
        main_layout = QVBoxLayout()

        # Label for the plugins layout
        main_layout.addWidget(QLabel('Plugins: '))
        self.models = ["* Text to image", "* Extend image", "* Edit image with text"]
        for i in range(len(self.models)):
            main_layout.addWidget(QLabel(self.models[i]))
        verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        main_layout.addItem(verticalSpacer)

        # Buttons layout
        button_layout = QHBoxLayout()
        login_button = QPushButton('Create account', self)
        login_button.clicked.connect(self.login_openai)
        see_usage_button = QPushButton('See credit usage', self)
        see_usage_button.clicked.connect(self.get_openai_usage)
        button_layout.addWidget(login_button)
        button_layout.addWidget(see_usage_button)
        main_layout.addItem(button_layout)
        
        # Set key layout
        key_layout = QHBoxLayout()
        key_layout.addWidget(QLabel("Key : "))
        self.openai_key = QPlainTextEdit()
        self.openai_key.setFixedWidth(450)
        self.openai_key.setFixedHeight(24)
        try:
            with open(os.path.join(os.path.dirname(__file__), "config.json"), "r") as json_file:
                key = json.load(json_file)['openai']['key']
        except:
            key = ""
        self.openai_key.appendPlainText(key)
        # self.text.resize(400, 5)
        self.openai_key = self.disable_plain_text(self.openai_key)        
        key_layout.addWidget(self.openai_key)
        main_layout.addItem(key_layout)

        # Update key layout
        self.update_key_button = QPushButton('Update key', self)
        main_layout.addWidget(self.update_key_button)
        self.update_key_button.clicked.connect(self.enable_plain_text)
        
        # Widget for video guide layout
        self.browser = QWebEngineView()
        main_layout.addWidget(self.browser)
        self.browser.setUrl(QUrl("https://www.youtube.com/embed/OB99E7Y1cMA"))        

        main_layout.addStretch(5)
        main.setLayout(main_layout)
        return main

    def update_key(self):
        val = self.openai_key.toPlainText()
        v = json.load(open(os.path.join(os.path.dirname(__file__), "config.json")))
        v['openai']['key'] = val
        with open(os.path.join(os.path.dirname(__file__), "config.json"), "w") as json_file:
            json.dump(v, json_file, indent=3)
        self.openai_key = self.disable_plain_text(self.openai_key) 
        self.update_key_button.setText("Update Key")
        self.update_key_button.clicked.connect(self.enable_plain_text)

    def enable_plain_text(self):
        self.openai_key.setReadOnly(False)
        # palette = self.openai_key.palette()
        # palette.setColor(QPalette.ColorRole.Text, QColor(240, 240, 240))  # Light grey 
        # palette.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))  # Grey text
        # self.openai_key.setPalette(palette)
        # self.openai_key.repaint()
        self.update_key_button.setText("Save Key")
        self.update_key_button.clicked.connect(self.update_key)
        # return text_widget

    def disable_plain_text(self, text_widget):
        text_widget.setReadOnly(True)
        # palette = text_widget.palette()
        # palette.setColor(QPalette.ColorRole.Text, QColor(240, 240, 240))  # Light grey 
        # palette.setColor(QPalette.ColorRole.Base, QColor(128, 128, 128))  # Grey text
        # text_widget.setPalette(palette)
        # self.openai_key.repaint()
        return text_widget


    def login_openai(self):
        webbrowser.open("https://platform.openai.com/signup")

    def get_openai_usage(self):
        webbrowser.open("https://platform.openai.com/settings/organization/billing/overview")

    def ui2(self):
        main_layout = QVBoxLayout()
        main_layout.addWidget(QLabel('page 2'))
        main_layout.addStretch(5)
        main = QWidget()
        main.setLayout(main_layout)
        return main

if __name__ == '__main__':

    app = QApplication(sys.argv)
    # app.setWindowIcon(QIcon(os.path.join(os.path.dirname(__file__), "images", "icon.png")))  # Set the application icon
    # Ensure the application keeps running even if the window is closed
    app.setQuitOnLastWindowClosed(False)
    ex = TrayWindow()
    ex.show()
    sys.exit(app.exec())