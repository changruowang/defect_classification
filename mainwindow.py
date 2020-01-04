# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QGraphicsPixmapItem, QGraphicsScene
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox
import sys
from glob import glob
from utils import *
from test import *

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowModality(QtCore.Qt.NonModal)
        MainWindow.resize(987, 595)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSpacing(7)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setObjectName("graphicsView")
        self.horizontalLayout.addWidget(self.graphicsView)

        self.label2 = QtWidgets.QLabel(self.centralwidget)
        self.label2.setGeometry(QtCore.QRect(15, 10, 181, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.label2.setFont(font)
        self.label2.setObjectName("label2")

        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setSpacing(3)
        self.verticalLayout.setObjectName("verticalLayout")
        self.treeWidget = QtWidgets.QTreeWidget(self.centralwidget)
        self.treeWidget.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.treeWidget.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.treeWidget.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.treeWidget.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked|QtWidgets.QAbstractItemView.EditKeyPressed|QtWidgets.QAbstractItemView.SelectedClicked)
        self.treeWidget.setObjectName("treeWidget")
    
        self.verticalLayout.addWidget(self.treeWidget)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setContentsMargins(0, -1, -1, 0)
        self.horizontalLayout_3.setSpacing(0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setObjectName("pushButton_4")
        self.horizontalLayout_3.addWidget(self.pushButton_4)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setEnabled(True)
        self.pushButton.setMouseTracking(False)
        self.pushButton.setIconSize(QtCore.QSize(20, 20))
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_3.addWidget(self.pushButton)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.verticalLayout.setStretch(0, 8)
        self.verticalLayout.setStretch(1, 1)
        self.verticalLayout.setStretch(2, 1)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.horizontalLayout.setStretch(0, 25)
        self.horizontalLayout.setStretch(1, 1)
        self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 987, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        
        self.treeWidget.clicked['QModelIndex'].connect(MainWindow.clickedTree)
        self.pushButton_4.clicked.connect(MainWindow.clickFileButton)
        self.pushButton.clicked.connect(MainWindow.clickRunButton)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # self.pix_item = QGraphicsPixmapItem()  
        # self.scene=QGraphicsScene()
        # self.scene.addItem(self.pix_item)
        # self.graphicsView.setScene(self.scene)
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.treeWidget.headerItem().setText(0, _translate("MainWindow", "name"))
        self.treeWidget.headerItem().setText(1, _translate("MainWindow", "score"))
        self.treeWidget.headerItem().setText(2, _translate("MainWindow", "class"))
        __sortingEnabled = self.treeWidget.isSortingEnabled()
        self.treeWidget.setSortingEnabled(False)

        self.treeWidget.setSortingEnabled(__sortingEnabled)
        self.pushButton_4.setText(_translate("MainWindow", "打开文件夹"))
        self.pushButton.setText(_translate("MainWindow", "开始"))
        self.pushButton.setEnabled(False)
        self.label.setText(_translate("MainWindow", ""))
        pa = QPalette()
        pa.setColor(pa.Foreground, QColor(255,0,0))
        self.label2.setPalette(pa)
        self.label2.setText(_translate("MainWindow", ""))
class load_model(QtCore.QThread):
    _signal_log = pyqtSignal(str)
    def __init__(self, obj, model_name='gauze'):
        super(load_model, self).__init__()
        self.model_name = model_name
        self.obj = obj
    def run(self):
        self._signal_log.emit('loading weights...')
        self.obj.load_weight(self.model_name)
        self._signal_log.emit('weights {:} OK'.format(self.model_name))
        
class test_one_img(QtCore.QThread):
    _signal_log = pyqtSignal(str)
    _signal_result = pyqtSignal(dict)
    def __init__(self, obj, parm):
        super(test_one_img, self).__init__()
        self.obj = obj
        if type(parm) != type([]):
            parm = [parm]
        self.parm = parm
    def run(self):
        results = {}
        file_names, _ = get_filenames(self.parm, with_ext=True)
        self._signal_log.emit('start classify {:d}%'.format(0))
        for idx, path in enumerate(self.parm):
            score, kind = self.obj.test_one_img(path)
            results[file_names[idx]] = [score, kind]
            self._signal_log.emit('classify {:d}%...'.format(int(idx/len(self.parm)*100.0)))
        self._signal_log.emit('classify OK!')
        self._signal_result.emit(results)


class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)
        self.tree_data = {}
        self.base_path = None
        self.select_imgname = None
        self._translate = QtCore.QCoreApplication.translate
        self.thread = None
        self.model = classify_obj()
        if not self.model.model_ready:
            self.thread = load_model(self.model)
            self.thread._signal_log.connect(self.call_backlog)
            self.thread.start()
    def clickedTree(self, currentindex):
        self.pushButton.setEnabled(True)
        self.select_imgname = self.treeWidget.currentItem().text(0)    
        img = QImage(os.path.join(self.base_path ,self.select_imgname))
        pix = QPixmap.fromImage(img)
        pix_item = QGraphicsPixmapItem(pix) 
        scene=QGraphicsScene() 
        #self.graphicsView.fitInView(self.pix_item)
        # self.scene.setSceneRect()
        scene.addItem(pix_item)
        self.graphicsView.setScene(scene)
        self.graphicsView.show()
        result = self.tree_data[self.select_imgname]
        self.change_result_show('{:}  {:}'.format(result[0], result[1]))

    def clearTreeItemAll(self): 
        items = self.treeWidget.topLevelItemCount()
        for idx in range(items):
            self.treeWidget.takeTopLevelItem(items - idx - 1)
    def addTreeItem(self, name, col1=None, col2=None):
        QtWidgets.QTreeWidgetItem(self.treeWidget)
        item = self.treeWidget.topLevelItem(self.treeWidget.topLevelItemCount()-1)
        item.setText(0, self._translate("MainWindow", name))
        if col1 is not None:
            item.setText(1, self._translate("MainWindow", str(col1)))
        if col2 is not None:
            item.setText(2, self._translate("MainWindow", str(col2)))
    def changeTreeShow(self):
        self.clearTreeItemAll()
        for k, v in self.tree_data.items():
            self.addTreeItem(k, v[0], v[1])
    def clickFileButton(self):
        file_name = QFileDialog.getExistingDirectory(self,"选择图片路径") 
        imgs_list = glob(file_name+'/*.jpg')
        if imgs_list:
            file_names, self.base_path = get_filenames(imgs_list, with_ext=True)
            for idx, name in enumerate(file_names):
                self.tree_data[name] = ['', '']
            self.changeTreeShow()
        else:
            QMessageBox.warning(self,"警告","该目录下无图片")    
            # QMessageBox.warning(self,"警告","该目录下无图片",QMessageBox.Yes | QMessageBox.No)        
    def clickRunButton(self):
        if not self.model.model_ready:
            return
        if self.select_imgname is not None:
            self.thread = test_one_img(self.model, os.path.join(self.base_path, self.select_imgname))
            self.thread._signal_result.connect(self.call_backresult)
            self.thread._signal_log.connect(self.call_backlog)
            self.thread.start()
    def call_backlog(self, msg):
        self.label.setText(self._translate("MainWindow", msg))
    def call_backresult(self, result):
        print(self.select_imgname)
        for k, v in result.items():
            self.tree_data[k] = ['{:.3f}'.format(v[0]), v[1]]
            self.change_result_show('{:3f}  {:}'.format(v[0], v[1]))
        self.changeTreeShow()
        
    def change_result_show(self, msg):
        self.label2.setText(self._translate("MainWindow", msg))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.show()
    sys.exit(app.exec_())