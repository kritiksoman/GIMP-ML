import json
from PyQt6 import sip
from PyQt6.QtWidgets import *
from PyQt6.QtGui import QPixmap
import sys, os
import requests
import webbrowser
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtGui import QPalette, QColor, QCursor
from PyQt6.QtGui import QAction
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import QUrl
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtGui import QClipboard
from PyQt6 import QtCore
import subprocess
import threading
import os
import socket
import win32com.client as win32




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
        # self.setWindowFlags(Qt.WindowType.Tool)
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowTitleHint | Qt.WindowType.CustomizeWindowHint | Qt.WindowType.WindowCloseButtonHint | Qt.WindowType.WindowMaximizeButtonHint | Qt.WindowType.Tool)
        self.setWindowIcon(QIcon(os.path.join(os.path.dirname(__file__), "images", "icon.png")))
        self.service_process = None

        # set the size of window
        self.Width = 800
        self.height = int(0.618 * self.Width)
        self.resize(self.Width, self.height)

        # add all widgets for the left and the right panel
        self.openai_button = QPushButton('OpenAI', self)
        self.settings_button = QPushButton('Settings', self)
        self.status = QLabel("Service Status: Starting.", self)
        

        self.openai_button.clicked.connect(self.activate_openai_panel)
        self.settings_button.clicked.connect(self.activate_settings_panel)

        self.models = list()

        # add tabs
        self.openai_tab = self.tab_openai()
        self.tab2 = self.tab_settings()

        self.initUI()


        # Set the window flag to hide from dock
        # self.setWindowFlags(Qt.WindowType.Tool)

        # System Tray Minimize
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(QIcon(os.path.join(os.path.dirname(__file__), "images", "icon.png")))
        self.tray_menu = QMenu() # Create the tray menu
        restore_action = QAction("Show window", self)
        restore_action.triggered.connect(self.show_raise_app)
        self.tray_menu.addAction(restore_action)

        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.quit_app)#QApplication.instance().quit)
        self.tray_menu.addAction(quit_action)

        self.tray_icon.setContextMenu(self.tray_menu)
        self.tray_icon.setVisible(True)
        if sys.platform == "win32": # Connect the tray icon click event for left click on windows
            self.tray_icon.activated.connect(self.on_tray_icon_activated) 

        # GIMP-ML fast api service
        self.identify_empty_port()
        service_thread = threading.Thread(target=self.test_start_service)
        self.progressSignal.connect(self.status.setText)
        service_thread.start()

    def on_tray_icon_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self.tray_menu.exec(QCursor.pos())
            
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

    def identify_empty_port(self):
        sock = socket.socket()
        sock.bind(('', 0))
        self.port = sock.getsockname()[1]
        v = json.load(open(os.path.join(os.path.dirname(__file__), "config.json")))
        v['gimpml']['port'] = self.port
        with open(os.path.join(os.path.dirname(__file__), "config.json"), "w") as json_file:
            json.dump(v, json_file, indent=3)


    def test_start_service(self):
        try:
            response = requests.get(r"http://localhost:"+ str(self.port) +"/status")
            response_data = response.json()
            if response_data['service'] == 'running':
                self.progressSignal.emit("Service Status: Running.")
        except:
            if sys.platform == "linux" or sys.platform == "linux2":
                self.service_process = subprocess.Popen([os.path.join(os.path.dirname(__file__),  "gimp_env", "bin", "python"), #TODO: add .. when running in  VSCODE
                os.path.join(os.path.dirname(__file__), "service.py")], 
                stdout=subprocess.PIPE, universal_newlines=True, shell=True)
            elif sys.platform == "darwin":
                # with open(r"/Users/kritiksoman/dev/tmp.txt", "w") as f:
                #     f.write(os.path.join(os.path.dirname(__file__)))
                self.service_process = subprocess.Popen([os.path.join(os.path.dirname(__file__), "gimp_env", "bin", "python"), #TODO: add .. when running in  VSCODE
                os.path.join(os.path.dirname(__file__), "service.py")], 
                stdout=subprocess.PIPE, universal_newlines=True, shell=False)
            elif sys.platform == "win32":
                self.service_process = subprocess.Popen([os.path.join(os.path.dirname(__file__),  "gimp_env", "Scripts", "python.exe"),#TODO: add .. when running in  VSCODE
                            os.path.join(os.path.dirname(__file__), "service.py")], 
                            stdout=subprocess.PIPE, universal_newlines=True, shell=True)
        while True:
            try:
                response = requests.get(r"http://localhost:"+ str(self.port) +"/status")
                response_data = response.json()
                if response_data['service'] == 'running':
                    self.progressSignal.emit("Service Status: Running.")
                    break
            except:
                pass
            
    def initUI(self):
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.openai_button)
        left_layout.addWidget(self.settings_button)
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

    def activate_settings_panel(self):
        self.right_widget.setCurrentIndex(1)

    def button3(self):
        self.right_widget.setCurrentIndex(2)

    def button4(self):
        self.right_widget.setCurrentIndex(3)

    def message(self, s):
        self.openai_key.appendPlainText(s)

    def tab_openai(self):
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
        login_button = QPushButton('Genarate API Key', self)
        login_button.clicked.connect(self.login_openai)
        see_usage_button = QPushButton('See credit usage', self)
        see_usage_button.clicked.connect(self.get_openai_usage)
        button_layout.addWidget(login_button)
        button_layout.addWidget(see_usage_button)
        main_layout.addItem(button_layout)
        
        # Set key layout
        key_layout = QHBoxLayout()
        label = QLabel("Key : ")
        label.setFixedWidth(100)
        key_layout.addWidget(label)
        self.openai_key = QPlainTextEdit()
        self.openai_key.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Plain)
        self.openai_key.setFixedWidth(550)
        self.openai_key.setFixedHeight(24)
        try:
            with open(os.path.join(os.path.dirname(__file__), "config.json"), "r") as json_file:
                key = json.load(json_file)['openai']['key']
        except:
            key = " "*51
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
        browser = QWebEngineView()
        main_layout.addWidget(browser)
        browser.setUrl(QUrl("https://www.youtube.com/embed/5LVX7MyI2Kg"))        

        # label = QLabel()
        # pixmap = QPixmap(r"D:\win\Users\Kritik Soman\Documents\GIMP-ML Assets\key gen.png")
        # label.setPixmap(pixmap)
        # label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # label.resize(600, 400)
        # main_layout.addWidget(label)
                     
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
        webbrowser.open("https://platform.openai.com/api-keys")

    def get_openai_usage(self):
        webbrowser.open("https://platform.openai.com/settings/organization/billing/overview")

    def tab_settings(self):
        main = QWidget()
        main_layout = QVBoxLayout()

        # Label for the plugins layout
        # main_layout.addWidget(QLabel('Plugin Path: '))
        # self.models = ["* Text to image", "* Extend image", "* Edit image with text"]
        # for i in range(len(self.models)):
        #     main_layout.addWidget(QLabel(self.models[i]))
        # verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        # main_layout.addItem(verticalSpacer)
        
        # self.plugin_path = QPlainTextEdit()
        # self.plugin_path.setFixedWidth(450)
        # self.plugin_path.setFixedHeight(24)
        # self.plugin_path.appendPlainText(os.path.join(os.path.dirname(__file__), "gimp2"))
        # main_layout.addWidget(self.plugin_path)
        
        # # Buttons layout
        # button_layout = QHBoxLayout()
        # login_button = QPushButton('Create account', self)
        # login_button.clicked.connect(self.login_openai)
        # see_usage_button = QPushButton('See credit usage', self)
        # see_usage_button.clicked.connect(self.get_openai_usage)
        # button_layout.addWidget(login_button)
        # button_layout.addWidget(see_usage_button)
        # main_layout.addItem(button_layout)
        
        # Plugin path layout
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("Plugin Path : "))
        self.plugin_path = QPlainTextEdit()
        self.plugin_path.setFixedWidth(450)
        self.plugin_path.setFixedHeight(24)
        self.plugin_path.appendPlainText(os.path.join(os.path.dirname(__file__), "gimp2"))  
        self.plugin_path.setReadOnly(True)
        path_layout.addWidget(self.plugin_path)
        main_layout.addItem(path_layout)

        # Update key layout
        self.copy_button = QPushButton('Copy', self)
        main_layout.addWidget(self.copy_button)
        self.copy_button.clicked.connect(self.copy_path)

        # Widget for video guide layout
        main_layout.addWidget(QLabel("Plugin not visible in GIMP? Follow this tutorial : "))
        browser = QWebEngineView()
        main_layout.addWidget(browser)
        browser.setUrl(QUrl("https://www.youtube.com/embed/cr0Fu0SY9eg"))     
        # label = QLabel()
        # pixmap = QPixmap(r"D:\win\Users\Kritik Soman\Documents\GIMP-ML Assets\add path.png")
        # label.setPixmap(pixmap)
        # label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # # label.resize(600, 400)
        # main_layout.addWidget(label)
        
        # Launch at startup layout
        startup_layout = QHBoxLayout()
        # startup_layout.addWidget(QLabel("Launch GIMP-ML at login : "))
        self.startup_checkbox = QCheckBox("Launch GIMP-ML at login", )
        startup_layout.addWidget(self.startup_checkbox)
        self.startup_checkbox.setChecked(self.is_present_at_startup())
        self.startup_checkbox.stateChanged.connect(self.update_startup)
        main_layout.addItem(startup_layout)   

        main_layout.addStretch(5)
        main.setLayout(main_layout)
        return main

    def update_startup(self):
        if self.startup_checkbox.isChecked():
            self.add_to_startup()
        else:
            self.remove_from_startup()
        
    @staticmethod
    def add_to_startup(program_name="GIMPML"):
        startup_path = os.path.join(os.getenv('APPDATA'), r'Microsoft\Windows\Start Menu\Programs\Startup')
        destination_path = os.path.join(startup_path, f"{program_name}.lnk")

        # Create a shortcut
        shell = win32.Dispatch("WScript.Shell")
        shortcut = shell.CreateShortCut(destination_path)
        shortcut.TargetPath = os.path.join(os.path.dirname(__file__), "GIMPML.exe")
        shortcut.save()
        
    @staticmethod
    def remove_from_startup(program_name="GIMPML"):
        startup_path = os.path.join(os.getenv('APPDATA'), r'Microsoft\Windows\Start Menu\Programs\Startup')
        destination_path = os.path.join(startup_path, f"{program_name}.lnk")
        if os.path.exists(destination_path):
            os.remove(destination_path)
            # print(f"Shortcut removed from {destination_path}")
        # else:
        #     print(f"No shortcut found at {destination_path}")
        
    @staticmethod
    def is_present_at_startup(program_name="GIMPML"):
        startup_path = os.path.join(os.getenv('APPDATA'), r'Microsoft\Windows\Start Menu\Programs\Startup')
        destination_path = os.path.join(startup_path, f"{program_name}.lnk")
        if os.path.exists(destination_path):
            return True
        return False
            
    def copy_path(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.plugin_path.toPlainText())
        

if __name__ == '__main__':
    if sys.platform == "darwin":
        import ctypes
        from Foundation import NSBundle
        # Hide the application from the macOS dock
        bundle = NSBundle.mainBundle()
        info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
        info['LSUIElement'] = '1'

    app = QApplication(sys.argv)
    # app.setWindowIcon(QIcon(os.path.join(os.path.dirname(__file__), "images", "icon.png")))  # Set the application icon
    # Ensure the application keeps running even if the window is closed
    app.setQuitOnLastWindowClosed(False)
    ex = TrayWindow()
    ex.show()
    sys.exit(app.exec())