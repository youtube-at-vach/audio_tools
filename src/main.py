
import argparse
import importlib
import pkgutil

from . import measurement_modules

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="A command-line tool for various audio measurements.")
    subparsers = parser.add_subparsers(dest="command", title="Available measurements", required=True)

    # measurement_modulesパッケージ内のモジュールを動的に読み込む
    for _, module_name, _ in pkgutil.iter_modules(measurement_modules.__path__):
        module = importlib.import_module(f".{module_name}", measurement_modules.__name__)
        for attribute_name in dir(module):
            attribute = getattr(module, attribute_name)
            if isinstance(attribute, type) and issubclass(attribute, measurement_modules.MeasurementModule) and attribute is not measurement_modules.MeasurementModule:
                instance = attribute()
                subparser = subparsers.add_parser(instance.name, help=instance.description)
                # ここで各モジュール固有の引数を追加することも可能
                # instance.add_arguments(subparser)

    args = parser.parse_args()

    # 対応するモジュールを探して実行
    for _, module_name, _ in pkgutil.iter_modules(measurement_modules.__path__):
        module = importlib.import_module(f".{module_name}", measurement_modules.__name__)
        for attribute_name in dir(module):
            attribute = getattr(module, attribute_name)
            if isinstance(attribute, type) and issubclass(attribute, measurement_modules.MeasurementModule) and attribute is not measurement_modules.MeasurementModule:
                instance = attribute()
                if instance.name == args.command:
                    instance.run(args)
                    return
