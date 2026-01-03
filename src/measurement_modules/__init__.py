
import abc


class MeasurementModule(abc.ABC):
    """
    全ての測定モジュールが継承する基底クラス。
    各モジュールは、名前(name)と説明(description)を持ち、
    runメソッドを実装する必要があります。
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """モジュールの名前（コマンドラインで指定する名前）"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """モジュールの説明（ヘルプメッセージに表示）"""
        raise NotImplementedError

    @abc.abstractmethod
    def run(self, args):
        """測定を実行するメソッド"""
        raise NotImplementedError
