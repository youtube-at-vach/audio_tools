import json
import os

from src.core.utils import resource_path


class LocalizationManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(LocalizationManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, 'initialized'):
            return
        self.language = 'en'
        self.translations = {}
        self.available_languages = {}
        self.initialized = True
        self.load_available_languages()

    def load_available_languages(self):
        # Scan assets/lang for json files
        self.available_languages = {}
        lang_dir = resource_path('src/assets/lang')
        if os.path.exists(lang_dir):
            for f in os.listdir(lang_dir):
                if f.endswith('.json'):
                    lang_code = f[:-5]
                    self.available_languages[lang_code] = f

        # Ensure en is always available (fallback)
        if 'en' not in self.available_languages:
             self.available_languages['en'] = 'en.json'

    def load_language(self, lang_code):
        if lang_code not in self.available_languages:
            # print(f"Language {lang_code} not found, falling back to en")
            lang_code = 'en'

        self.language = lang_code
        path = resource_path(f'src/assets/lang/{lang_code}.json')

        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    self.translations = json.load(f)
            except Exception as e:
                print(f"Failed to load language {lang_code}: {e}")
                self.translations = {}
        else:
            # If en.json doesn't exist, we just use keys
            self.translations = {}

    def get(self, key, default=None):
        val = self.translations.get(key)
        if val is None:
            return default if default is not None else key
        return val

# Global instance
_loc_manager = LocalizationManager()

def tr(key, default=None):
    return _loc_manager.get(key, default)

def get_manager():
    return _loc_manager
