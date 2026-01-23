"""Configuration dialogs for interrogator models."""

import logging
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QComboBox, QDoubleSpinBox, QPushButton,
                             QFormLayout, QGroupBox, QCheckBox, QLineEdit,
                             QTabWidget, QWidget, QProgressBar, QMessageBox,
                             QListWidget, QListWidgetItem, QScrollArea)
from PyQt6.QtCore import Qt
from typing import Dict
from pathlib import Path
from core.clip_model_loader import get_categorized_models
from core import FileManager
from ui.workers import OrganizationWorker

logger = logging.getLogger(__name__)


def _populate_clip_models_combo(clip_model_combo: QComboBox):
    """Populate CLIP model combo box with categorized models."""
    try:
        # Get categorized models
        models_dict = get_categorized_models()

        current_idx = 0

        # SD 1.x Models
        if models_dict['sd_1x']:
            clip_model_combo.addItem("=== SD 1.x Models (Recommended) ===")
            clip_model_combo.model().item(current_idx).setEnabled(False)
            current_idx += 1

            for model in models_dict['sd_1x']:
                display_text = model
                if model == 'ViT-L-14/openai':
                    display_text = f"{model} (Default)"
                clip_model_combo.addItem(display_text)
                clip_model_combo.setItemData(current_idx, model, Qt.ItemDataRole.UserRole)
                current_idx += 1

            clip_model_combo.insertSeparator(current_idx)
            current_idx += 1

        # SD 2.0 Models
        if models_dict['sd_20']:
            clip_model_combo.addItem("=== SD 2.0 Models ===")
            clip_model_combo.model().item(current_idx).setEnabled(False)
            current_idx += 1

            for model in models_dict['sd_20']:
                clip_model_combo.addItem(model)
                clip_model_combo.setItemData(current_idx, model, Qt.ItemDataRole.UserRole)
                current_idx += 1

            clip_model_combo.insertSeparator(current_idx)
            current_idx += 1

        # SDXL Models
        if models_dict['sdxl']:
            clip_model_combo.addItem("=== SDXL Models ===")
            clip_model_combo.model().item(current_idx).setEnabled(False)
            current_idx += 1

            for model in models_dict['sdxl']:
                display_text = model
                if model == 'ViT-bigG-14/laion2b_s39b_b160k':
                    display_text = f"{model} (SDXL Default)"
                clip_model_combo.addItem(display_text)
                clip_model_combo.setItemData(current_idx, model, Qt.ItemDataRole.UserRole)
                current_idx += 1

            clip_model_combo.insertSeparator(current_idx)
            current_idx += 1

        # Other Models
        if models_dict['other']:
            clip_model_combo.addItem("=== Other Models ===")
            clip_model_combo.model().item(current_idx).setEnabled(False)
            current_idx += 1

            for model in models_dict['other']:
                clip_model_combo.addItem(model)
                clip_model_combo.setItemData(current_idx, model, Qt.ItemDataRole.UserRole)
                current_idx += 1

        logger.info(
            f"Loaded CLIP models: SD1.x={len(models_dict['sd_1x'])}, "
            f"SD2.0={len(models_dict['sd_20'])}, SDXL={len(models_dict['sdxl'])}, "
            f"Other={len(models_dict['other'])}"
        )

    except Exception as e:
        logger.error(f"Error populating CLIP models: {e}")
        # Fallback to minimal list
        clip_model_combo.addItems([
            'ViT-L-14/openai',
            'ViT-H-14/laion2b_s32b_b79k',
            'ViT-g-14/laion2b_s12b_b42k',
            'ViT-B-32/openai',
            'ViT-B-16/openai'
        ])
        logger.info("Using fallback CLIP model list")


def _select_first_valid_clip_model(clip_model_combo: QComboBox):
    """Select the first selectable (non-header) model in the combo box."""
    for i in range(clip_model_combo.count()):
        item = clip_model_combo.model().item(i)
        if item and item.isEnabled():
            clip_model_combo.setCurrentIndex(i)
            return
    # Fallback to index 0 if no enabled items found
    clip_model_combo.setCurrentIndex(0)


def create_clip_config_widget(clip_config: Dict, parent=None) -> tuple:
    """
    Create CLIP configuration widget with all settings.

    Args:
        clip_config: Dictionary with CLIP configuration
        parent: Parent widget

    Returns:
        Tuple of (widget, references_dict) where references_dict contains:
        {
            'clip_model_combo': QComboBox,
            'caption_model_combo': QComboBox,
            'mode_combo': QComboBox,
            'device_combo': QComboBox
        }
    """
    widget = QWidget(parent)
    layout = QVBoxLayout(widget)

    # Form layout for settings
    form_layout = QFormLayout()

    # CLIP Model selection
    clip_model_combo = QComboBox()
    _populate_clip_models_combo(clip_model_combo)
    current_clip_model = clip_config.get('clip_model', 'ViT-L-14/openai')
    index = clip_model_combo.findText(current_clip_model)
    if index >= 0:
        clip_model_combo.setCurrentIndex(index)
    else:
        _select_first_valid_clip_model(clip_model_combo)
    form_layout.addRow("CLIP Model:", clip_model_combo)

    # Caption Model selection
    caption_model_combo = QComboBox()
    caption_model_combo.addItems([
        'None',
        'blip-base',
        'blip-large',
        'blip2-2.7b',
        'blip2-flan-t5-xl',
        'git-large-coco'
    ])
    current_caption_model = clip_config.get('caption_model', 'None')
    index = caption_model_combo.findText(current_caption_model if current_caption_model else 'None')
    if index >= 0:
        caption_model_combo.setCurrentIndex(index)
    form_layout.addRow("Caption Model:", caption_model_combo)

    # Mode selection
    clip_mode_combo = QComboBox()
    clip_mode_combo.addItems(['best', 'fast', 'classic', 'negative'])
    current_mode = clip_config.get('mode', 'best')
    clip_mode_combo.setCurrentText(current_mode)
    form_layout.addRow("Interrogation Mode:", clip_mode_combo)

    # Device selection
    clip_device_combo = QComboBox()
    clip_device_combo.addItems(['cuda', 'cpu'])
    current_device = clip_config.get('device', 'cuda')
    clip_device_combo.setCurrentText(current_device)
    form_layout.addRow("Device:", clip_device_combo)

    layout.addLayout(form_layout)

    # Model descriptions
    model_desc_group = QGroupBox("CLIP Model Information")
    model_desc_layout = QVBoxLayout()
    model_desc_layout.addWidget(QLabel(
        "CLIP Models:\n"
        "• ViT-L-14: Large model, good balance\n"
        "• ViT-H-14: Huge model, best quality\n"
        "• ViT-g-14: Giant model, highest quality\n"
        "• ViT-B-32/16: Base models, faster"
    ))
    model_desc_group.setLayout(model_desc_layout)
    layout.addWidget(model_desc_group)

    # Caption model descriptions
    caption_desc_group = QGroupBox("Caption Model Information")
    caption_desc_layout = QVBoxLayout()
    caption_desc_layout.addWidget(QLabel(
        "Caption Models (requires clip-interrogator >= 0.6.0):\n"
        "• None: Use CLIP only\n"
        "• blip-base/large: BLIP models\n"
        "• blip2-2.7b: BLIP2, improved quality\n"
        "• blip2-flan-t5-xl: BLIP2 with T5, best quality\n"
        "• git-large-coco: GIT model"
    ))
    caption_desc_group.setLayout(caption_desc_layout)
    layout.addWidget(caption_desc_group)

    # Mode descriptions
    mode_desc_group = QGroupBox("Mode Descriptions")
    mode_desc_layout = QVBoxLayout()
    mode_desc_layout.addWidget(QLabel(
        "• best: Highest quality, slowest\n"
        "• fast: Quick processing, good quality\n"
        "• classic: Traditional approach\n"
        "• negative: Generate negative prompts"
    ))
    mode_desc_group.setLayout(mode_desc_layout)
    layout.addWidget(mode_desc_group)

    layout.addStretch()

    # Return widget and references to controls
    references = {
        'clip_model_combo': clip_model_combo,
        'caption_model_combo': caption_model_combo,
        'mode_combo': clip_mode_combo,
        'device_combo': clip_device_combo
    }

    return widget, references


def create_wd_config_widget(wd_config: Dict, parent=None) -> tuple:
    """
    Create WD Tagger configuration widget with all settings.

    Args:
        wd_config: Dictionary with WD configuration
        parent: Parent widget

    Returns:
        Tuple of (widget, references_dict) where references_dict contains:
        {
            'wd_model_combo': QComboBox,
            'threshold_spin': QDoubleSpinBox,
            'device_combo': QComboBox
        }
    """
    widget = QWidget(parent)
    layout = QVBoxLayout(widget)

    # Form layout for settings
    form_layout = QFormLayout()

    # WD Model selection
    wd_model_combo = QComboBox()
    wd_model_combo.addItems([
        # V1.4 models
        'SmilingWolf/wd-v1-4-moat-tagger-v2',
        'SmilingWolf/wd-v1-4-vit-tagger-v2',
        'SmilingWolf/wd-v1-4-vit-tagger',
        'SmilingWolf/wd-v1-4-convnext-tagger-v2',
        'SmilingWolf/wd-v1-4-convnext-tagger',
        'SmilingWolf/wd-v1-4-convnextv2-tagger-v2',
        'SmilingWolf/wd-v1-4-swinv2-tagger-v2',
        # V3 models (latest)
        'SmilingWolf/wd-vit-tagger-v3',
        'SmilingWolf/wd-vit-large-tagger-v3',
        'SmilingWolf/wd-convnext-tagger-v3',
        'SmilingWolf/wd-swinv2-tagger-v3',
        'SmilingWolf/wd-eva02-large-tagger-v3'
    ])
    current_wd_model = wd_config.get('wd_model', 'SmilingWolf/wd-v1-4-moat-tagger-v2')
    index = wd_model_combo.findText(current_wd_model)
    if index >= 0:
        wd_model_combo.setCurrentIndex(index)
    form_layout.addRow("WD Model:", wd_model_combo)

    # Threshold
    threshold_spin = QDoubleSpinBox()
    threshold_spin.setRange(0.0, 1.0)
    threshold_spin.setSingleStep(0.05)
    threshold_spin.setDecimals(2)
    threshold_spin.setValue(wd_config.get('threshold', 0.35))
    form_layout.addRow("Confidence Threshold:", threshold_spin)

    # Device selection
    wd_device_combo = QComboBox()
    wd_device_combo.addItems(['cuda', 'cpu'])
    current_device = wd_config.get('device', 'cuda')
    wd_device_combo.setCurrentText(current_device)
    form_layout.addRow("Device:", wd_device_combo)

    layout.addLayout(form_layout)

    # Model descriptions
    model_desc_group = QGroupBox("WD Model Information")
    model_desc_layout = QVBoxLayout()
    model_desc_layout.addWidget(QLabel(
        "V1.4 Models (Stable):\n"
        "• moat-tagger-v2: MOAT architecture (recommended for v1.4)\n"
        "• vit-tagger-v2/vit-tagger: Vision Transformer variants\n"
        "• convnext-tagger-v2/convnext-tagger: ConvNeXt architecture variants\n"
        "• convnextv2-tagger-v2: ConvNeXt V2 architecture\n"
        "• swinv2-tagger-v2: Swin Transformer V2\n"
        "\n"
        "V3 Models (Latest - Recommended):\n"
        "• vit-tagger-v3: Vision Transformer (improved)\n"
        "• vit-large-tagger-v3: Large ViT model (best quality)\n"
        "• convnext-tagger-v3: ConvNeXt (improved)\n"
        "• swinv2-tagger-v3: Swin Transformer V2 (improved)\n"
        "• eva02-large-tagger-v3: EVA-02 Large (highest quality)"
    ))
    model_desc_group.setLayout(model_desc_layout)
    layout.addWidget(model_desc_group)

    # Threshold description
    threshold_desc_group = QGroupBox("Threshold Information")
    threshold_desc_layout = QVBoxLayout()
    threshold_desc_layout.addWidget(QLabel(
        "The confidence threshold determines which tags are included.\n"
        "• Lower values (0.2-0.3): More tags, less precise\n"
        "• Medium values (0.35-0.5): Balanced (recommended)\n"
        "• Higher values (0.5-0.8): Fewer tags, more confident"
    ))
    threshold_desc_group.setLayout(threshold_desc_layout)
    layout.addWidget(threshold_desc_group)

    layout.addStretch()

    # Return widget and references to controls
    references = {
        'wd_model_combo': wd_model_combo,
        'threshold_spin': threshold_spin,
        'device_combo': wd_device_combo
    }

    return widget, references


def create_camie_config_widget(camie_config: Dict, parent=None) -> tuple:
    """
    Create Camie Tagger configuration widget with all settings.

    Args:
        camie_config: Dictionary with Camie configuration
        parent: Parent widget

    Returns:
        Tuple of (widget, references_dict) where references_dict contains:
        {
            'camie_model_combo': QComboBox,
            'threshold_spin': QDoubleSpinBox,
            'threshold_profile_combo': QComboBox,
            'device_combo': QComboBox,
            'category_checkboxes': Dict[str, QCheckBox],
            'category_threshold_spins': Dict[str, QDoubleSpinBox],
            'category_thresholds_group': QGroupBox
        }
    """
    widget = QWidget(parent)
    layout = QVBoxLayout(widget)

    # Form layout for main settings
    form_layout = QFormLayout()

    # Camie Model selection
    camie_model_combo = QComboBox()
    camie_model_combo.addItems([
        'Camais03/camie-tagger-v2',
        'Camais03/camie-tagger'
    ])
    current_model = camie_config.get('camie_model', 'Camais03/camie-tagger-v2')
    index = camie_model_combo.findText(current_model)
    if index >= 0:
        camie_model_combo.setCurrentIndex(index)
    form_layout.addRow("Camie Model:", camie_model_combo)

    # Threshold profile
    threshold_profile_combo = QComboBox()
    threshold_profile_combo.addItems([
        'overall',
        'micro_optimized',
        'macro_optimized',
        'balanced',
        'category_specific'
    ])
    current_profile = camie_config.get('threshold_profile', 'overall')
    threshold_profile_combo.setCurrentText(current_profile)
    form_layout.addRow("Threshold Profile:", threshold_profile_combo)

    # Base threshold
    threshold_spin = QDoubleSpinBox()
    threshold_spin.setRange(0.0, 1.0)
    threshold_spin.setSingleStep(0.05)
    threshold_spin.setDecimals(2)
    threshold_spin.setValue(camie_config.get('threshold', 0.5))
    form_layout.addRow("Base Threshold:", threshold_spin)

    # Device selection
    camie_device_combo = QComboBox()
    camie_device_combo.addItems(['cuda', 'cpu'])
    current_device = camie_config.get('device', 'cuda')
    camie_device_combo.setCurrentText(current_device)
    form_layout.addRow("Device:", camie_device_combo)

    layout.addLayout(form_layout)

    # Category filter group
    category_filter_group = QGroupBox("Category Filters")
    category_filter_layout = QVBoxLayout()

    category_info = QLabel(
        "Select which tag categories to include in output.\n"
        "Unchecked categories will be excluded from results."
    )
    category_info.setWordWrap(True)
    category_info.setStyleSheet("QLabel { font-size: 9pt; color: #666; }")
    category_filter_layout.addWidget(category_info)

    # Category checkboxes
    categories = ['general', 'character', 'copyright', 'artist', 'meta', 'rating', 'year']
    enabled_categories = camie_config.get('enabled_categories', categories.copy())
    category_checkboxes = {}

    category_checkbox_layout = QHBoxLayout()
    for category in categories:
        cb = QCheckBox(category.capitalize())
        cb.setChecked(category in enabled_categories)
        category_checkboxes[category] = cb
        category_checkbox_layout.addWidget(cb)

    category_filter_layout.addLayout(category_checkbox_layout)
    category_filter_group.setLayout(category_filter_layout)
    layout.addWidget(category_filter_group)

    # Category-specific thresholds group (collapsible)
    category_thresholds_group = QGroupBox("Category-Specific Thresholds")
    category_thresholds_group.setCheckable(True)
    category_thresholds_group.setChecked(current_profile == 'category_specific')
    category_thresholds_layout = QFormLayout()

    # Get default category thresholds
    default_cat_thresholds = {
        'artist': 0.5, 'character': 0.5, 'copyright': 0.5,
        'general': 0.35, 'meta': 0.5, 'rating': 0.5, 'year': 0.5
    }
    config_cat_thresholds = camie_config.get('category_thresholds', default_cat_thresholds)

    category_threshold_spins = {}
    for category in categories:
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 1.0)
        spin.setSingleStep(0.05)
        spin.setDecimals(2)
        spin.setValue(config_cat_thresholds.get(category, 0.5))
        category_threshold_spins[category] = spin
        category_thresholds_layout.addRow(f"{category.capitalize()}:", spin)

    category_thresholds_group.setLayout(category_thresholds_layout)
    layout.addWidget(category_thresholds_group)

    # Show/hide category thresholds based on profile
    def on_profile_changed(profile_text):
        is_category_specific = profile_text == 'category_specific'
        category_thresholds_group.setVisible(is_category_specific)

    threshold_profile_combo.currentTextChanged.connect(on_profile_changed)
    # Set initial visibility
    category_thresholds_group.setVisible(current_profile == 'category_specific')

    # Model descriptions
    model_desc_group = QGroupBox("Camie Model Information")
    model_desc_layout = QVBoxLayout()
    model_desc_layout.addWidget(QLabel(
        "Camie Tagger Models:\n"
        "• camie-tagger-v2: Latest version (recommended)\n"
        "• camie-tagger: Original version (v1)\n\n"
        "Features ~70,527 tags across 7 categories:\n"
        "Artist, Character, Copyright, General, Meta, Rating, Year"
    ))
    model_desc_group.setLayout(model_desc_layout)
    layout.addWidget(model_desc_group)

    # Threshold profile descriptions
    profile_desc_group = QGroupBox("Threshold Profile Information")
    profile_desc_layout = QVBoxLayout()
    profile_desc_layout.addWidget(QLabel(
        "Profiles determine how thresholds are applied:\n"
        "• Overall: Single threshold for all categories (0.5)\n"
        "• Micro Optimized: More tags, better recall (0.614)\n"
        "• Macro Optimized: Balanced performance (0.492)\n"
        "• Balanced: Middle ground (0.55)\n"
        "• Category-Specific: Set different thresholds per category"
    ))
    profile_desc_group.setLayout(profile_desc_layout)
    layout.addWidget(profile_desc_group)

    layout.addStretch()

    # Return widget and references to controls
    references = {
        'camie_model_combo': camie_model_combo,
        'threshold_spin': threshold_spin,
        'threshold_profile_combo': threshold_profile_combo,
        'device_combo': camie_device_combo,
        'category_checkboxes': category_checkboxes,
        'category_threshold_spins': category_threshold_spins,
        'category_thresholds_group': category_thresholds_group
    }

    return widget, references
  

class ModelConfigDialog(QDialog):
    """Unified configuration dialog for CLIP and WD models with tabs."""

    def __init__(self, parent=None, clip_config: Dict = None, wd_config: Dict = None):
        super().__init__(parent)
        self.setWindowTitle("Model Configuration")
        self.setModal(True)
        self.clip_config = clip_config or {}
        self.wd_config = wd_config or {}
        self.setup_ui()

    def setup_ui(self):
        """Setup the UI components with tabs."""
        layout = QVBoxLayout(self)

        # Create tab widget
        self.tabs = QTabWidget()

        # CLIP tab
        clip_tab = self.create_clip_tab()
        self.tabs.addTab(clip_tab, "CLIP")

        # WD tab
        wd_tab = self.create_wd_tab()
        self.tabs.addTab(wd_tab, "WD Tagger")

        layout.addWidget(self.tabs)

        # Buttons
        button_layout = QHBoxLayout()

        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        button_layout.addWidget(ok_button)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)

        self.setMinimumWidth(500)

    def create_clip_tab(self) -> QWidget:
        """Create CLIP configuration tab using shared widget creator."""
        widget, references = create_clip_config_widget(self.clip_config, self)

        # Store references to controls for later access
        self.clip_model_combo = references['clip_model_combo']
        self.caption_model_combo = references['caption_model_combo']
        self.clip_mode_combo = references['mode_combo']
        self.clip_device_combo = references['device_combo']

        return widget

    def create_wd_tab(self) -> QWidget:
        """Create WD Tagger configuration tab using shared widget creator."""
        widget, references = create_wd_config_widget(self.wd_config, self)

        # Store references to controls for later access
        self.wd_model_combo = references['wd_model_combo']
        self.threshold_spin = references['threshold_spin']
        self.wd_device_combo = references['device_combo']

        return widget

    def get_clip_config(self) -> Dict:
        """Get CLIP configuration."""
        caption_model = self.caption_model_combo.currentText()

        # Get actual model string from UserRole data (not display text)
        current_idx = self.clip_model_combo.currentIndex()
        clip_model = self.clip_model_combo.itemData(current_idx, Qt.ItemDataRole.UserRole)

        # Fallback: strip decorations if UserRole data not set
        if clip_model is None:
            clip_model = self.clip_model_combo.currentText()
            clip_model = clip_model.replace(' (Default)', '').replace(' (SDXL Default)', '')

        return {
            'clip_model': clip_model,
            'caption_model': None if caption_model == 'None' else caption_model,
            'mode': self.clip_mode_combo.currentText(),
            'device': self.clip_device_combo.currentText()
        }

    def get_wd_config(self) -> Dict:
        """Get WD configuration."""
        return {
            'wd_model': self.wd_model_combo.currentText(),
            'threshold': self.threshold_spin.value(),
            'device': self.wd_device_combo.currentText()
        }

    def get_config(self) -> Dict:
        """Get both configurations."""
        return {
            'clip': self.get_clip_config(),
            'wd': self.get_wd_config()
        }


class CLIPConfigDialog(QDialog):
    """Legacy CLIP configuration dialog - kept for compatibility."""

    def __init__(self, parent=None, current_config: Dict = None):
        super().__init__(parent)
        self.setWindowTitle("CLIP Configuration")
        self.setModal(True)
        self.config = current_config or {}
        self.setup_ui()

    def setup_ui(self):
        """Setup the UI components."""
        layout = QVBoxLayout(self)

        # Form layout for settings
        form_layout = QFormLayout()

        # Mode selection
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['best', 'fast', 'classic', 'negative'])
        current_mode = self.config.get('mode', 'best')
        self.mode_combo.setCurrentText(current_mode)
        form_layout.addRow("Interrogation Mode:", self.mode_combo)

        # Device selection
        self.device_combo = QComboBox()
        self.device_combo.addItems(['cuda', 'cpu'])
        current_device = self.config.get('device', 'cuda')
        self.device_combo.setCurrentText(current_device)
        form_layout.addRow("Device:", self.device_combo)

        layout.addLayout(form_layout)

        # Mode descriptions
        desc_group = QGroupBox("Mode Descriptions")
        desc_layout = QVBoxLayout()
        desc_layout.addWidget(QLabel(
            "• best: Highest quality, slowest\n"
            "• fast: Quick processing, good quality\n"
            "• classic: Traditional approach\n"
            "• negative: Generate negative prompts"
        ))
        desc_group.setLayout(desc_layout)
        layout.addWidget(desc_group)

        # Buttons
        button_layout = QHBoxLayout()

        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        button_layout.addWidget(ok_button)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)

    def get_config(self) -> Dict:
        """Get the configured settings."""
        return {
            'mode': self.mode_combo.currentText(),
            'device': self.device_combo.currentText()
        }


class WDConfigDialog(QDialog):
    """Legacy WD configuration dialog - kept for compatibility."""

    def __init__(self, parent=None, current_config: Dict = None):
        super().__init__(parent)
        self.setWindowTitle("WD Tagger Configuration")
        self.setModal(True)
        self.config = current_config or {}
        self.setup_ui()

    def setup_ui(self):
        """Setup the UI components."""
        layout = QVBoxLayout(self)

        # Form layout for settings
        form_layout = QFormLayout()

        # Threshold
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setValue(self.config.get('threshold', 0.35))
        form_layout.addRow("Confidence Threshold:", self.threshold_spin)

        # Device selection
        self.device_combo = QComboBox()
        self.device_combo.addItems(['cuda', 'cpu'])
        current_device = self.config.get('device', 'cuda')
        self.device_combo.setCurrentText(current_device)
        form_layout.addRow("Device:", self.device_combo)

        layout.addLayout(form_layout)

        # Threshold description
        desc_group = QGroupBox("Threshold Information")
        desc_layout = QVBoxLayout()
        desc_layout.addWidget(QLabel(
            "The confidence threshold determines which tags are included.\n"
            "• Lower values (0.2-0.3): More tags, less precise\n"
            "• Medium values (0.35-0.5): Balanced (recommended)\n"
            "• Higher values (0.5-0.8): Fewer tags, more confident"
        ))
        desc_group.setLayout(desc_layout)
        layout.addWidget(desc_group)

        # Buttons
        button_layout = QHBoxLayout()

        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        button_layout.addWidget(ok_button)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)

    def get_config(self) -> Dict:
        """Get the configured settings."""
        return {
            'threshold': self.threshold_spin.value(),
            'device': self.device_combo.currentText()
        }


class OrganizeDialog(QDialog):
    """Dialog for organizing images by tags with progress tracking."""

    def __init__(self, current_directory: str = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Organize Images by Tags")
        self.setModal(True)
        self.resize(600, 500)

        self.current_directory = Path(current_directory) if current_directory else None
        self.available_tags = []
        self.organization_worker = None
        self.moved_count = 0
        self.subdirectories = []  # List of subdirectories found

        self.setup_ui()

        # Load available tags if directory is provided
        if self.current_directory:
            self._load_available_tags()

    def setup_ui(self):
        """Setup the UI components."""
        layout = QVBoxLayout(self)

        # Directory info
        self.dir_label = QLabel("No directory selected")
        self.dir_label.setWordWrap(True)
        self.dir_label.setStyleSheet("QLabel { font-weight: bold; }")
        layout.addWidget(self.dir_label)

        if self.current_directory:
            self.dir_label.setText(f"Directory: {self.current_directory}")

        # Warning label
        warning_label = QLabel(
            "⚠️ WARNING: This operation will MOVE files from their current location to a subdirectory. "
            "Files will be removed from their original location."
        )
        warning_label.setWordWrap(True)
        warning_label.setStyleSheet(
            "QLabel { "
            "font-weight: bold; "
            "color: #d32f2f; "
            "background-color: #ffebee; "
            "padding: 10px; "
            "border: 2px solid #d32f2f; "
            "border-radius: 4px; "
            "}"
        )
        layout.addWidget(warning_label)

        # Form layout
        form_layout = QFormLayout()

        # Tag criteria
        self.tags_edit = QLineEdit()
        self.tags_edit.setPlaceholderText("tag1, tag2, tag3")
        form_layout.addRow("Tags to match:", self.tags_edit)

        # Target subdirectory
        self.subdir_edit = QLineEdit()
        self.subdir_edit.setPlaceholderText("organized")
        self.subdir_edit.setText("organized")
        form_layout.addRow("Target subdirectory:", self.subdir_edit)

        # Match mode
        self.match_mode_combo = QComboBox()
        self.match_mode_combo.addItems(['any', 'all'])
        form_layout.addRow("Match mode:", self.match_mode_combo)

        # Move text files
        self.move_text_check = QCheckBox("Move .txt files with images")
        self.move_text_check.setChecked(True)
        form_layout.addRow("", self.move_text_check)

        # Recursive checkbox
        self.recursive_check = QCheckBox("Include subdirectories (recursive)")
        self.recursive_check.setChecked(False)
        self.recursive_check.stateChanged.connect(self._on_recursive_changed)
        form_layout.addRow("", self.recursive_check)

        layout.addLayout(form_layout)

        # Directory selection (initially hidden)
        self.dir_selection_group = QGroupBox("Select Source Directories")
        self.dir_selection_group.setVisible(False)
        dir_selection_layout = QVBoxLayout()

        dir_info = QLabel(
            "Choose which subdirectories to include as source for organization.\n"
            "This prevents reorganizing already-organized files."
        )
        dir_info.setWordWrap(True)
        dir_info.setStyleSheet("QLabel { font-size: 9pt; color: #666; }")
        dir_selection_layout.addWidget(dir_info)

        # Directory list with checkboxes
        self.dir_list = QListWidget()
        self.dir_list.setMaximumHeight(150)
        dir_selection_layout.addWidget(self.dir_list)

        # Select/Deselect all buttons
        dir_buttons_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self._select_all_directories)
        dir_buttons_layout.addWidget(self.select_all_btn)

        self.deselect_all_btn = QPushButton("Deselect All")
        self.deselect_all_btn.clicked.connect(self._deselect_all_directories)
        dir_buttons_layout.addWidget(self.deselect_all_btn)

        dir_selection_layout.addLayout(dir_buttons_layout)
        self.dir_selection_group.setLayout(dir_selection_layout)
        layout.addWidget(self.dir_selection_group)

        # Info
        info_label = QLabel(
            "Match mode:\n"
            "• any: Image has at least one matching tag\n"
            "• all: Image has all specified tags"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Available tags display
        self.tags_label = QLabel("")
        self.tags_label.setWordWrap(True)
        layout.addWidget(self.tags_label)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMinimumHeight(30)
        self.progress_bar.setStyleSheet(
            "QProgressBar {"
            "border: 2px solid #2196F3;"
            "border-radius: 5px;"
            "text-align: center;"
            "font-weight: bold;"
            "}"
            "QProgressBar::chunk {"
            "background-color: #2196F3;"
            "}"
        )
        layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("QLabel { font-weight: bold; color: #1976D2; }")
        layout.addWidget(self.progress_label)

        # Buttons
        button_layout = QHBoxLayout()

        self.organize_button = QPushButton("Move Images")
        self.organize_button.clicked.connect(self._start_organization)
        button_layout.addWidget(self.organize_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._cancel_organization)
        self.cancel_button.setEnabled(False)
        button_layout.addWidget(self.cancel_button)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.reject)
        button_layout.addWidget(self.close_button)

        layout.addLayout(button_layout)

    def _load_available_tags(self):
        """Load available tags from the current directory."""
        if not self.current_directory:
            return

        self.available_tags = list(FileManager.get_all_tags_in_directory(str(self.current_directory)))

        if self.available_tags:
            tags_preview = ", ".join(sorted(self.available_tags)[:20])
            if len(self.available_tags) > 20:
                tags_preview += f"... ({len(self.available_tags) - 20} more)"
            self.tags_label.setText(f"\nAvailable tags ({len(self.available_tags)}): {tags_preview}")
        else:
            self.tags_label.setText("\nNo tagged images found in directory")

    def _on_recursive_changed(self):
        """Handle recursive checkbox state change."""
        if self.recursive_check.isChecked():
            # Show directory selection and scan for subdirectories
            self._scan_subdirectories()
            self.dir_selection_group.setVisible(True)
        else:
            # Hide directory selection
            self.dir_selection_group.setVisible(False)

    def _scan_subdirectories(self):
        """Scan for subdirectories in the current directory."""
        if not self.current_directory:
            return

        self.dir_list.clear()
        self.subdirectories = []

        # Get all subdirectories (non-recursive scan)
        for item in sorted(self.current_directory.iterdir()):
            if item.is_dir():
                # Get relative path
                rel_path = item.relative_to(self.current_directory)
                self.subdirectories.append(str(rel_path))

                # Add to list with checkbox
                list_item = QListWidgetItem(str(rel_path))
                list_item.setFlags(list_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                list_item.setCheckState(Qt.CheckState.Checked)  # Default to checked
                self.dir_list.addItem(list_item)

        # Also add root directory option
        root_item = QListWidgetItem("(Root directory)")
        root_item.setFlags(root_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        root_item.setCheckState(Qt.CheckState.Checked)
        root_item.setData(Qt.ItemDataRole.UserRole, ".")  # Store "." as marker for root
        self.dir_list.insertItem(0, root_item)

    def _select_all_directories(self):
        """Select all directory checkboxes."""
        for i in range(self.dir_list.count()):
            item = self.dir_list.item(i)
            item.setCheckState(Qt.CheckState.Checked)

    def _deselect_all_directories(self):
        """Deselect all directory checkboxes."""
        for i in range(self.dir_list.count()):
            item = self.dir_list.item(i)
            item.setCheckState(Qt.CheckState.Unchecked)

    def _get_selected_directories(self):
        """Get list of selected directories."""
        selected = []
        for i in range(self.dir_list.count()):
            item = self.dir_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                # Check if it's the root directory
                marker = item.data(Qt.ItemDataRole.UserRole)
                if marker == ".":
                    selected.append(".")
                else:
                    selected.append(item.text())
        return selected

    def _start_organization(self):
        """Start the organization process."""
        if not self.current_directory:
            QMessageBox.warning(self, "Warning", "No directory selected")
            return

        # Get tags
        tags_text = self.tags_edit.text().strip()
        if not tags_text:
            QMessageBox.warning(self, "Warning", "Please specify tags to match")
            return

        tags = [tag.strip() for tag in tags_text.split(',') if tag.strip()]

        # Get subdirectory
        subdir = self.subdir_edit.text().strip() or 'organized'

        # Get match mode
        match_mode = self.match_mode_combo.currentText()

        # Get move text option
        move_text = self.move_text_check.isChecked()

        # Get recursive option
        recursive = self.recursive_check.isChecked()
        selected_dirs = []

        # Get images
        if recursive:
            # Get selected directories
            selected_dirs = self._get_selected_directories()

            if not selected_dirs:
                QMessageBox.warning(self, "Warning", "Please select at least one source directory")
                return

            # Find images recursively
            all_images = FileManager.find_images(str(self.current_directory), recursive=True)

            # Filter images to only include those from selected directories
            images = []
            for img in all_images:
                img_path = Path(img)
                # Get relative path from current directory
                try:
                    rel_path = img_path.relative_to(self.current_directory)
                    # Check if image is in one of the selected directories
                    if "." in selected_dirs:
                        # Root directory is selected, check if image is directly in root
                        if len(rel_path.parts) == 1:
                            images.append(img)
                            continue

                    # Check if image is in any of the selected subdirectories
                    for selected_dir in selected_dirs:
                        if selected_dir != "." and str(rel_path).startswith(selected_dir):
                            images.append(img)
                            break
                except ValueError:
                    # Image is not relative to current directory, skip it
                    continue
        else:
            # Non-recursive: only current directory
            images = FileManager.find_images(str(self.current_directory), recursive=False)

        if not images:
            QMessageBox.warning(self, "Warning", "No images found in selected directories")
            return

        # Confirmation dialog
        tags_str = ", ".join(tags)
        move_text_str = "Yes" if move_text else "No"
        recursive_str = "Yes" if recursive else "No"

        confirmation_msg = (
            f"⚠️ WARNING: This will MOVE files (not copy)!\n\n"
            f"Operation Details:\n"
            f"• Images to process: {len(images)}\n"
            f"• Tags to match: {tags_str}\n"
            f"• Match mode: {match_mode}\n"
            f"• Target subdirectory: {subdir}\n"
            f"• Move .txt files: {move_text_str}\n"
            f"• Recursive search: {recursive_str}\n"
        )

        if recursive and selected_dirs:
            dirs_str = ", ".join(selected_dirs)
            confirmation_msg += f"• Source directories: {dirs_str}\n"

        confirmation_msg += (
            f"\nFiles will be permanently moved from their current location to:\n"
            f"{self.current_directory / subdir}\n\n"
            f"Do you want to proceed?"
        )

        reply = QMessageBox.question(
            self,
            "Confirm Move Operation",
            confirmation_msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No  # Default to No for safety
        )

        if reply == QMessageBox.StandardButton.No:
            return

        # Start worker
        self.organization_worker = OrganizationWorker(
            images,
            tags,
            subdir,
            match_mode,
            move_text
        )

        # Connect signals
        self.organization_worker.progress.connect(self._on_progress)
        self.organization_worker.moved.connect(self._on_image_moved)
        self.organization_worker.error.connect(self._on_error)
        self.organization_worker.finished.connect(self._on_finished)

        # Update UI
        self.organize_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.close_button.setEnabled(False)
        self.progress_bar.setMaximum(len(images))
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.progress_label.setText(f"Starting move operation... (0/{len(images)})")
        self.moved_count = 0

        # Start
        self.organization_worker.start()

    def _cancel_organization(self):
        """Cancel the organization process."""
        if self.organization_worker:
            self.organization_worker.cancel()
            self.progress_label.setText("Cancelling...")

    def _on_progress(self, current: int, total: int, message: str):
        """Handle organization progress."""
        self.progress_bar.setValue(current)
        percentage = int((current / total) * 100) if total > 0 else 0
        self.progress_label.setText(f"Processing: {current}/{total} ({percentage}%) - {message}")

    def _on_image_moved(self, source: str, dest: str):
        """Handle image moved."""
        self.moved_count += 1

    def _on_error(self, image_path: str, error: str):
        """Handle organization error."""
        logger.error(f"Error organizing {image_path}: {error}")

    def _on_finished(self, moved_count: int):
        """Handle organization completion."""
        self.progress_label.setText(f"Organization complete - moved {moved_count} images")
        self.organize_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.close_button.setEnabled(True)
        self.progress_bar.setVisible(False)

        QMessageBox.information(
            self,
            "Complete",
            f"Moved {moved_count} images to subdirectory '{self.subdir_edit.text()}'"
        )
