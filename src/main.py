import argparse
import importlib
import pkgutil
import inquirer
import sys

from . import measurement_modules
from .measurement_modules.base import MeasurementModule

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="A command-line tool for various audio measurements.")
    
    # measurement_modulesパッケージ内のモジュールを動的に読み込む
    modules = []
    for _, module_name, _ in pkgutil.iter_modules(measurement_modules.__path__):
        module = importlib.import_module(f".{module_name}", measurement_modules.__name__)
        for attribute_name in dir(module):
            attribute = getattr(module, attribute_name)
            if isinstance(attribute, type) and issubclass(attribute, MeasurementModule) and attribute is not MeasurementModule:
                modules.append(attribute())

    # コマンドライン引数でモジュールが指定されているか確認
    parser.add_argument("command", nargs='?', default=None, help="The measurement module to run.")
    args = parser.parse_args()

    command = args.command

    # コマンドが指定されていない場合は、対話的に選択
    if command is None:
        questions = [
            inquirer.List('command',
                            message="どの測定ツールを実行しますか？",
                            choices=[module.name for module in modules],
                            ),
        ]
        answers = inquirer.prompt(questions)
        if answers is None:
            return
        command = answers['command']

    # 対応するモジュールを探して実行
    for module in modules:
        if module.name == command:
            # ここで、選択されたモジュールが引数を必要とする場合は、
            # さらに入力を求めるなどの処理を追加できます。
            # この例では、引数なしでrunメソッドを呼び出します。
            module.run(argparse.Namespace()) 
            return

if __name__ == '__main__':
    main()
