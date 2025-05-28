import csv
import numpy as np
from rich.console import Console
import os # For checking if directory exists in example

# Matplotlib is an optional dependency for generate_plot
try:
    import matplotlib.pyplot as plt
    import matplotlib
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    # Provide a dummy plt structure if needed for type hinting or unconditional calls,
    # but the current design checks MATPLOTLIB_AVAILABLE before using plt.
    class DummyPlt:
        def figure(self, **kwargs): return DummyFig()
        def plot(self, *args, **kwargs): pass
        def legend(self, **kwargs): pass
        def xlabel(self, *args, **kwargs): pass
        def ylabel(self, *args, **kwargs): pass
        def title(self, *args, **kwargs): pass
        def grid(self, *args, **kwargs): pass
        def xscale(self, *args, **kwargs): pass
        def yscale(self, *args, **kwargs): pass
        def savefig(self, *args, **kwargs): pass
        def show(self, **kwargs): pass
        def close(self, *args, **kwargs): pass
    class DummyFig:
        def add_subplot(self, *args, **kwargs): return DummyAx()
    class DummyAx: # Minimal Axes methods if used directly
        def plot(self, *args, **kwargs): pass
        def legend(self, **kwargs): pass
        def set_xlabel(self, *args, **kwargs): pass
        def set_ylabel(self, *args, **kwargs): pass
        def set_title(self, *args, **kwargs): pass
        def grid(self, *args, **kwargs): pass
        def set_xscale(self, *args, **kwargs): pass
        def set_yscale(self, *args, **kwargs): pass

    if not MATPLOTLIB_AVAILABLE:
        plt = DummyPlt() # Assign dummy if import failed


def save_results_to_csv(filepath: str, data_rows_list_of_dicts: list[dict], fieldnames: list[str], console: Console = None):
    """
    Writes a list of dictionaries to a CSV file.

    Args:
        filepath (str): The path to the CSV file.
        data_rows_list_of_dicts (list[dict]): Data to write. Each dict is a row.
        fieldnames (list[str]): List of keys to use for header and dict lookup, defining column order.
        console (Console, optional): Rich Console for printing messages.
    """
    effective_console = console if console else Console(stderr=True)
    
    if not data_rows_list_of_dicts:
        effective_console.print(f"[yellow]Warning: No data provided to save to CSV '{filepath}'. File not created.[/yellow]")
        return

    try:
        # Ensure directory exists
        dir_name = os.path.dirname(filepath)
        if dir_name: # If filepath includes a directory
            os.makedirs(dir_name, exist_ok=True)

        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore') # ignore extra keys in dicts
            writer.writeheader()
            for row_dict in data_rows_list_of_dicts:
                writer.writerow(row_dict)
        effective_console.print(f"[green]Successfully saved results to CSV: {filepath}[/green]")
    except IOError as e:
        effective_console.print(f"[bold red]IOError: Could not write to CSV file '{filepath}'. Error: {e}[/bold red]")
    except Exception as e:
        effective_console.print(f"[bold red]An unexpected error occurred while saving CSV to '{filepath}': {e}[/bold red]")


def generate_plot(
    x_data: list | np.ndarray,
    y_data_list: list[list | np.ndarray],
    legend_labels_list: list[str],
    title: str,
    x_label: str,
    y_label: str,
    output_filename: str = None,
    show_plot: bool = True,
    log_x_scale: bool = False,
    log_y_scale: bool = False,
    console: Console = None
):
    """
    Generates a plot using Matplotlib, with options to save and show.

    Args:
        x_data (list | np.ndarray): X-axis data.
        y_data_list (list[list | np.ndarray]): List of Y-axis datasets.
        legend_labels_list (list[str]): Labels for each Y-axis dataset.
        title (str): Plot title.
        x_label (str): X-axis label.
        y_label (str): Y-axis label.
        output_filename (str, optional): Filepath to save the plot. Defaults to None (don't save).
        show_plot (bool, optional): Whether to display the plot. Defaults to True.
        log_x_scale (bool, optional): Use logarithmic scale for X-axis. Defaults to False.
        log_y_scale (bool, optional): Use logarithmic scale for Y-axis. Defaults to False.
        console (Console, optional): Rich Console for printing messages.
    """
    effective_console = console if console else Console(stderr=True)

    if not MATPLOTLIB_AVAILABLE:
        effective_console.print("[bold red]Matplotlib is not installed. Cannot generate plot.[/bold red]")
        if output_filename:
            effective_console.print(f"[yellow]Plotting skipped. File '{output_filename}' will not be created.[/yellow]")
        return

    if not y_data_list or not any(y_data_list):
        effective_console.print("[yellow]Warning: No Y-data provided for plotting. Plot will be empty.[/yellow]")
        # Optionally, still create an empty plot with labels if desired
        # return 

    if len(y_data_list) != len(legend_labels_list):
        effective_console.print("[bold red]Error: Mismatch between number of Y-datasets and legend labels. Cannot generate plot.[/bold red]")
        return

    fig = None # Initialize fig to ensure it's defined for finally block
    try:
        # Ensure backend is appropriate, especially if not showing plot / running in headless env
        if not show_plot and output_filename:
            current_backend = matplotlib.get_backend()
            # List of common non-interactive backends
            non_interactive_backends = ['agg', 'cairo', 'pdf', 'ps', 'svg', 'template']
            if current_backend.lower() not in non_interactive_backends:
                try:
                    matplotlib.use('Agg') # Attempt to switch to a non-interactive backend
                except Exception as e_backend:
                    effective_console.print(f"[yellow]Warning: Could not switch to 'Agg' backend for non-interactive plotting: {e_backend}. Using '{current_backend}'.[/yellow]")


        fig = plt.figure(figsize=(10, 6)) # Create a new figure
        ax = fig.add_subplot(1, 1, 1)

        x_array = np.asarray(x_data) # Ensure x_data is array-like

        for i, y_data_raw in enumerate(y_data_list):
            y_array = np.asarray(y_data_raw)
            if len(x_array) != len(y_array) and len(y_array) > 0 : # Allow empty y_array to not plot
                effective_console.print(f"[yellow]Warning: Length mismatch for dataset '{legend_labels_list[i]}' (X: {len(x_array)}, Y: {len(y_array)}). Skipping this dataset.[/yellow]")
                continue
            
            if len(y_array) == 0:
                effective_console.print(f"[yellow]Warning: Dataset '{legend_labels_list[i]}' is empty. Skipping.[/yellow]")
                continue

            # Check for NaNs - Matplotlib typically handles them by not drawing segments
            # If we want to explicitly warn or skip:
            if np.all(np.isnan(y_array)):
                effective_console.print(f"[yellow]Warning: Dataset '{legend_labels_list[i]}' contains only NaNs. Skipping.[/yellow]")
                continue
            
            ax.plot(x_array, y_array, label=legend_labels_list[i])

        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        
        if any(y_data_list) and any(len(y)>0 for y in y_data_list if isinstance(y, (list, np.ndarray))): # Only show legend if there's something plotted
             if any(label for label in legend_labels_list): # And if there are labels
                ax.legend()

        ax.grid(True, which="both", ls="-", alpha=0.5)

        if log_x_scale:
            ax.set_xscale('log')
        if log_y_scale:
            ax.set_yscale('log')
        
        if output_filename:
            # Ensure directory exists
            dir_name = os.path.dirname(output_filename)
            if dir_name: # If filepath includes a directory
                os.makedirs(dir_name, exist_ok=True)
            try:
                plt.savefig(output_filename)
                effective_console.print(f"[green]Plot saved to {output_filename}[/green]")
            except Exception as e_save:
                effective_console.print(f"[bold red]Error saving plot to '{output_filename}': {e_save}[/bold red]")

        if show_plot:
            try:
                plt.show()
            except Exception as e_show:
                effective_console.print(f"[bold red]Error showing plot: {e_show}. This might happen in headless environments or if no display is available.[/bold red]")
                effective_console.print("[yellow]Try setting show_plot=False and providing an output_filename if you are in a non-GUI environment.[/yellow]")
                
    except Exception as e:
        effective_console.print(f"[bold red]An unexpected error occurred during plotting: {e}[/bold red]")
    finally:
        if MATPLOTLIB_AVAILABLE and fig is not None:
            # Close the figure to free resources, regardless of whether it was shown or saved.
            # This is important in scripts that generate many plots.
            plt.close(fig)


if __name__ == '__main__':
    console = Console()
    console.rule("[bold cyan]Testing common_audio_lib.output_formatting_utils[/bold cyan]")

    # --- Test save_results_to_csv ---
    console.print("\n[bold]1. Testing save_results_to_csv[/bold]")
    test_csv_dir = "test_outputs" # Create a subdir for test files
    test_csv_file = os.path.join(test_csv_dir, "test_results.csv")
    
    sample_data = [
        {'frequency': 100, 'level_db': -20.5, 'distortion_percent': 0.1},
        {'frequency': 200, 'level_db': -22.1, 'distortion_percent': 0.05, 'extra_col': 'ignored'},
        {'frequency': 500, 'level_db': -19.8, 'distortion_percent': 0.12},
    ]
    field_names = ['frequency', 'level_db', 'distortion_percent']
    
    # Test with data
    save_results_to_csv(test_csv_file, sample_data, field_names, console)
    try:
        with open(test_csv_file, 'r') as f:
            console.print(f"\nContents of '{test_csv_file}':")
            console.print(f.read())
    except FileNotFoundError:
        console.print(f"[red]File '{test_csv_file}' not found after trying to save.[/red]")

    # Test with empty data
    empty_csv_file = os.path.join(test_csv_dir, "empty_results.csv")
    save_results_to_csv(empty_csv_file, [], field_names, console)


    # --- Test generate_plot ---
    console.print("\n[bold]2. Testing generate_plot[/bold]")
    if not MATPLOTLIB_AVAILABLE:
        console.print("[yellow]Matplotlib not available, skipping generate_plot tests that create files/show plots.[/yellow]")
    else:
        test_plot_dir = "test_outputs" # Can be same as CSV dir
        test_plot_file = os.path.join(test_plot_dir, "test_plot.png")

        x = np.linspace(0, 2 * np.pi, 100)
        y1 = np.sin(x)
        y2 = np.cos(x)
        y_empty = []
        y_nans = np.full_like(x, np.nan)


        # Test 1: Basic plot, save and show
        console.print("\nTest 2.1: Basic plot (sin, cos), save and show")
        generate_plot(
            x_data=x,
            y_data_list=[y1, y2],
            legend_labels_list=['Sine Wave', 'Cosine Wave'],
            title='Sine and Cosine Waves',
            x_label='Angle (radians)',
            y_label='Amplitude',
            output_filename=test_plot_file,
            show_plot=False, # Set to False for automated tests to prevent blocking
            console=console
        )
        if os.path.exists(test_plot_file):
             console.print(f"Plot saved to {test_plot_file}. Please check it manually (if show_plot=False).")
        else:
             console.print(f"[red]Plot file {test_plot_file} was not created.[/red]")


        # Test 2: Logarithmic scale
        console.print("\nTest 2.2: Logarithmic scale plot")
        log_x = np.logspace(1, 3, 100) # 10^1 to 10^3
        log_y1 = 1 / log_x
        log_y2 = 1 / (log_x**2)
        log_plot_file = os.path.join(test_plot_dir, "log_plot.png")
        generate_plot(
            x_data=log_x,
            y_data_list=[log_y1, log_y2],
            legend_labels_list=['1/x', '1/x^2'],
            title='Logarithmic Scale Example',
            x_label='X (log scale)',
            y_label='Y (log scale)',
            log_x_scale=True,
            log_y_scale=True,
            output_filename=log_plot_file,
            show_plot=False, # False for automation
            console=console
        )

        # Test 3: Handling empty/NaN data
        console.print("\nTest 2.3: Plot with empty and NaN data")
        mixed_data_plot_file = os.path.join(test_plot_dir, "mixed_data_plot.png")
        generate_plot(
            x_data=x,
            y_data_list=[y1, y_empty, y_nans, y2],
            legend_labels_list=['Sine', 'Empty Data', 'NaN Data', 'Cosine'],
            title='Handling Special Data Cases',
            x_label='X',
            y_label='Y',
            output_filename=mixed_data_plot_file,
            show_plot=False, # False for automation
            console=console
        )
        
        # Test 4: Mismatch y_data and labels
        console.print("\nTest 2.4: Mismatched Y-data and labels (expected error)")
        generate_plot(
            x_data=x,
            y_data_list=[y1],
            legend_labels_list=['Label1', 'Label2'], # Mismatch
            title='Mismatch Test',
            x_label='X',
            y_label='Y',
            show_plot=False,
            console=console
        )
        
        # Test 5: No y_data
        console.print("\nTest 2.5: No Y-data (expected warning, empty plot)")
        no_data_plot_file = os.path.join(test_plot_dir, "no_data_plot.png")
        generate_plot(
            x_data=x,
            y_data_list=[],
            legend_labels_list=[],
            title='No Y-Data Test',
            x_label='X',
            y_label='Y',
            output_filename=no_data_plot_file,
            show_plot=False,
            console=console
        )

    console.rule("[bold cyan]End of tests[/bold cyan]")
