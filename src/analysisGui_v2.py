#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 14:32:48 2024

@author: danielvalmassei
"""

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QComboBox, QCheckBox, QLineEdit

class AnalysisApp(QWidget):
    def __init__(self):
        super().__init__()

        self.data_folder = None
        self.run_names = []
        self.data_channels = []
        self.noise_channel = None
        self.atten = 1.0
        self.trigger_threshold = 500
        self.signal_threshold = 200
        self.nbins = 128
        self.hist_endpoint = 20

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Data folder selection
        self.folder_label = QLabel('Data Folder:')
        layout.addWidget(self.folder_label)
        self.folder_button = QPushButton('Select Data Folder')
        self.folder_button.clicked.connect(self.selectFolder)
        layout.addWidget(self.folder_button)

        # Run names
        self.run_label = QLabel('Run Names (comma-separated):')
        layout.addWidget(self.run_label)
        self.run_edit = QLineEdit()
        layout.addWidget(self.run_edit)

        # Data channels
        self.channel_label = QLabel('Data Channels:')
        layout.addWidget(self.channel_label)
        self.channel_checkboxes = []
        for i in range(8):
            checkbox = QCheckBox(f'ch. {i}')
            self.channel_checkboxes.append(checkbox)
            layout.addWidget(checkbox)

        # Noise channel dropdown
        self.noise_label = QLabel('Noise Channel:')
        layout.addWidget(self.noise_label)
        self.noise_combo = QComboBox()
        for i in range(8):
            self.noise_combo.addItem(f'ch. {i}')
        layout.addWidget(self.noise_combo)

        # Other parameters
        params = {
            'Attenuation/Gain': self.atten,
            'Trigger Threshold': self.trigger_threshold,
            'Signal Threshold': self.signal_threshold,
            'Number of Bins': self.nbins,
            'Histogram Endpoint': self.hist_endpoint
        }
        for param, default_val in params.items():
            label = QLabel(f'{param}:')
            layout.addWidget(label)
            edit = QLineEdit(str(default_val))
            edit.setObjectName(param.lower().replace(' ', '_'))
            layout.addWidget(edit)

        # Run button
        self.run_button = QPushButton('Run Analysis')
        self.run_button.clicked.connect(self.runAnalysis)
        layout.addWidget(self.run_button)

        self.setLayout(layout)
        self.setWindowTitle('Analysis Application')

    def selectFolder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Data Folder')
        if folder:
            self.data_folder = folder
            self.folder_label.setText(f'Data Folder: {folder}')

    def runAnalysis(self):
        # Read input values
        self.run_names = [name.strip() for name in self.run_edit.text().split(',')]
        self.data_channels = [i for i, checkbox in enumerate(self.channel_checkboxes) if checkbox.isChecked()]
        self.noise_channel = self.noise_combo.currentIndex()
        self.atten = float(self.findChild(QLineEdit, 'attenuation').text())
        self.trigger_threshold = int(self.findChild(QLineEdit, 'trigger_threshold').text())
        self.signal_threshold = int(self.findChild(QLineEdit, 'signal_threshold').text())
        self.nbins = int(self.findChild(QLineEdit, 'number_of_bins').text())
        self.hist_endpoint = int(self.findChild(QLineEdit, 'histogram_endpoint').text())

        # Run analysis
        import analysis_lib
        analysis_lib.main(self.data_folder, self.run_names, self.data_channels, [self.noise_channel],
                          self.atten, self.trigger_threshold, self.signal_threshold, self.nbins, self.hist_endpoint)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AnalysisApp()
    ex.show()
    sys.exit(app.exec_())
