import argparse
import configparser
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.animation import FuncAnimation

class DistortionVisualizer:
    __version__ = "1.4"  # バージョン番号を更新

    def __init__(self, csv_file, zlabel='THD(dBr)', cal_file='calibration_settings.ini', amplitude_type='Output(dBFS)'):
        """
        DistortionVisualizerをCSVファイルを用いて初期化します。
        
        Parameters:
        csv_file : str
            'frequency', 'amplitude', 'distortion'列を含むCSVファイルのパス。
        cal_file : str, optional
            校正データが含まれているINIファイルのパス。デフォルトは 'calibration_settings.ini'。
        amplitude_type : str, optional
            振幅として使用するカラム。'Output(dBFS)' または 'Input(dBFS)'。
        """
        self.csv_file = csv_file
        self.cal_file = cal_file
        self.amplitude_type = amplitude_type
        self.data = None
        self.frequencies = None
        self.amplitudes = None
        self.distortion = None
        self.xlabel = amplitude_type
        self.ylabel = 'Frequency(Hz)'
        self.zlabel = zlabel
        self.output_conversion = None
        self.input_conversion = None
        self.load_calibration()
        self.read_and_validate()
    
    def load_calibration(self):
        """
        校正設定をファイルから読み込みます。
        """
        config = configparser.ConfigParser()
        
        try:
            config.read(self.cal_file)
            self.output_conversion = config.getfloat('Calibration', 'output_conversion')
            self.input_conversion = config.getfloat('Calibration', 'input_conversion')
            last_calibration_date = config.get('Calibration', 'last_calibration_date')
            print(f"Calibration data loaded (Last Calibration: {last_calibration_date})")
            print("--convert_to_dBVrmsオプションをつければ、dBVrms表示に変えられます。")
        except (FileNotFoundError, configparser.NoSectionError, configparser.NoOptionError):
            print(f"{self.cal_file} が見つかりません。校正データは使用できません。")
            self.output_conversion = None
            self.input_conversion = None
    
    def read_and_validate(self):
        """
        CSVファイルを読み込み、データを検証します。
        """
        if not os.path.isfile(self.csv_file):
            raise FileNotFoundError(f"CSVファイル '{self.csv_file}' が存在しません。")
        
        try:
            self.data = pd.read_csv(self.csv_file)
        except Exception as e:
            raise ValueError(f"CSVファイルの読み込みエラー: {e}")
        
        # 必要なカラムの確認
        required_columns = {'Frequency(Hz)', self.amplitude_type, self.zlabel}
        if not required_columns.issubset(self.data.columns):
            missing = required_columns - set(self.data.columns)
            raise ValueError(f"欠けているカラム: {missing}")

        # データのピボット
        try:
            pivot_table = self.data.pivot(index='Frequency(Hz)', columns=self.amplitude_type, values=self.zlabel)
        except ValueError:
            print("データが揃っていないため、補完が必要です。")
            self.interpolate_data()
            return
        
        # 欠損値の確認
        if pivot_table.isnull().values.any():
            print("ピボットされたデータに欠損値があるため、補完が適用されます。")
            self.interpolate_data()
            return
        
        self.frequencies = pivot_table.index.values
        self.amplitudes = pivot_table.columns.values
        self.distortion = pivot_table.values
    
    def interpolate_data(self):
        """
        データが揃っていない場合に、補完を行います。
        """
        # 元データの取得
        frequencies = self.data['Frequency(Hz)'].values
        amplitudes = self.data[self.amplitude_type].values
        distortions = self.data[self.zlabel].values
        
        # 等間隔のグリッドを作成（補完のため）
        grid_frequencies = np.logspace(np.log10(frequencies.min()), np.log10(frequencies.max()), num=100)
        grid_amplitudes = np.linspace(amplitudes.min(), amplitudes.max(), num=100)
        
        # 補完の実行
        grid_distortion = griddata((amplitudes, frequencies), distortions, (grid_amplitudes[None, :], grid_frequencies[:, None]), method='linear')
        
        self.frequencies = grid_frequencies
        self.amplitudes = grid_amplitudes
        self.distortion = grid_distortion

        print("警告: データが揃っていないため、補完されたデータを使用してプロットします。")
    
    def convert_amplitude_to_dBVrms(self):
        """
        振幅をdBVrmsに変換します。
        """
        if self.amplitude_type == 'Output(dBFS)' and self.output_conversion is not None:
            self.amplitudes = self.amplitudes + self.output_conversion
            self.xlabel = 'Output Amplitude(dBVrms)'  # 軸ラベルを更新
        elif self.amplitude_type == 'Input(dBFS)' and self.input_conversion is not None:
            self.amplitudes = self.amplitudes + self.input_conversion
            self.xlabel = 'Input Amplitude(dBVrms)'  # 軸ラベルを更新
        else:
            print("校正データが読み込まれていないため、振幅の変換は行われません。")
    
    def plot_contour(self, device_name='', cmap='viridis', title='Distortion Map', xlabel=None, ylabel='Frequency(Hz)', color=True):
        """
        歪みデータを等高線プロットとして描画します。
        
        Parameters:
        device_name : str, optional
            タイトルに含めるデバイス名。
        cmap : str, optional
            プロットに使用するカラーマップ。デフォルトは'viridis'。
        title : str, optional
            プロットのタイトル。
        xlabel : str, optional
            x軸のラベル。指定がない場合は現在のxlabelを使用。
        ylabel : str, optional
            y軸のラベル。
        """
        plt.figure(figsize=(10, 8))

        # タイトルにデバイス名を追加
        if device_name:
            plt.title(f'Distortion Visualizer (v{self.__version__}) - {title} - Device: {device_name}')
        else:
            plt.title(f'Distortion Visualizer (v{self.__version__}) - {title}')
        
        xlabel = xlabel or self.xlabel
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # x軸を線形スケール（振幅）
        plt.xscale('linear')

        # y軸を対数スケール（周波数）
        plt.yscale('log')

        # x軸の制限を設定
        plt.xlim(min(self.amplitudes), max(self.amplitudes))
        
        # -140から-20までの12の等高線レベルを作成
        contour_levels = np.arange(-140, -19, 5)
        
        if color:
            # 塗りつぶした等高線をプロット
            contour = plt.contourf(self.amplitudes, self.frequencies, self.distortion, levels=24, cmap=cmap)
            cbar = plt.colorbar(contour)
            cbar.set_label(self.zlabel)

        contour_lines = plt.contour(self.amplitudes, self.frequencies, self.distortion, levels=contour_levels, colors='black', linewidths=2)
        
        # 等高線のラベルを追加
        plt.clabel(contour_lines, inline=True, fontsize=10, fmt='%.1f dBr')

        plt.tight_layout()
        plt.show()

    def plot_3d(self, device_name='', cmap='viridis', title='Distortion Map 3D', animate=False):
        """
        歪みデータを3Dサーフェスプロットとして表示します。
        
        パラメータ:
        device_name : str, 任意
            タイトルに含めるデバイス名。
        cmap : str, 任意
            プロットに使用するカラーマップ（デフォルトは 'viridis'）。
        title : str, 任意
            プロットのタイトル。
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # If device name is given, add it to the title
        if device_name:
            ax.set_title(f'Distortion Visualizer (v{self.__version__}) - {title} - Device: {device_name}')
        else:
            ax.set_title(f'Distortion Visualizer (v{self.__version__}) - {title}')

        # Set axis labels
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_zlabel(self.zlabel)

        # Set axis scales
        ax.set_xscale('linear')

        # X, Y meshgrid for 3D plotting
        X, Y = np.meshgrid(self.amplitudes, np.log10(self.frequencies))
        Z = self.distortion

        # Plot the 3D surface
        surf = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='k', linewidth=0.5, alpha=0.7)
        fig.colorbar(surf, ax=ax, label=self.zlabel)

        # Y ticks
        tick_indices = np.linspace(0, len(self.frequencies) - 1, num=5).astype(int)
        plt.yticks(np.log10(self.frequencies[tick_indices]), [f"{int(self.frequencies[i])} Hz" for i in tick_indices])

        def update_view(num):
            ax.view_init(elev=ax.elev, azim=num)  # elevは高さ、azimは方位を設定
            return fig,

        plt.tight_layout()

        if animate:
            FuncAnimation(fig, update_view, frames=np.arange(-180, 180, 2), interval=300)  # 360度を2度刻みで回転
            plt.show()
        else:
            plt.show()


# 使用例
if __name__ == "__main__":
    # 引数パーサーを作成
    parser = argparse.ArgumentParser(description='CSVファイルから歪みデータを視覚化するツール。')
    parser.add_argument('csv_file', type=str, help='歪み測定データが含まれるCSVファイルへのパス。')
    parser.add_argument('-d', '--device_name', type=str, default='', help='タイトルに表示するデバイス名（任意）。')
    parser.add_argument('-a', '--amplitude_type', type=str, choices=['Output(dBFS)', 'Input(dBFS)'], default='Output(dBFS)', help='使用する振幅データの種類: "Output(dBFS)" または "Input(dBFS)"。')
    parser.add_argument('-c', '--convert_to_dBVrms', action='store_true', help='振幅をdBVrmsに変換する（校正データが必要）。')
    parser.add_argument('-p', '--plot_type', type=str, choices=['contour', '3d'], default='contour', help='グラフの種類を選択: "contour" (等高線) または "3d" (3Dサーフェス)。デフォルトは等高線。')
    parser.add_argument('--color', action='store_true', help='等高線プロットで色を表示する（白黒の代わりにカラーマップを使用）。')
    parser.add_argument('--rotate', action='store_true', help='3Dプロットを自動で回転させる（アニメーション）。')

    # 引数を解析
    args = parser.parse_args()
    
    # データ可視化の実行
    try:
        visualizer = DistortionVisualizer(args.csv_file, zlabel='THD(dBr)', amplitude_type=args.amplitude_type)
        
        # dBVrmsに変換する場合
        if args.convert_to_dBVrms:
            visualizer.convert_amplitude_to_dBVrms()
        
        if args.plot_type == 'contour':
            visualizer.plot_contour(device_name=args.device_name, color=args.color)
        elif args.plot_type == '3d':
            visualizer.plot_3d(device_name=args.device_name, animate=args.rotate)

    except Exception as e:
        print(f"エラー: {e}")
