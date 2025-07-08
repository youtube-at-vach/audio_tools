# Audio Loopback Finder

This tool automatically detects active loopback paths between output and input channels of a specified audio device.

## Description

When performing audio measurements, it is often necessary to create a "loopback" connection, where an output signal from an audio interface is fed back into one of its inputs. This tool simplifies the process of verifying which channel combination constitutes a valid loopback path.

It works by sending a short test tone out of each available output channel, one by one, while simultaneously recording on all available input channels. It then analyzes the recorded audio to see if the test tone is present, and if so, reports the corresponding output -> input channel pair as a valid loopback path.

## Dependencies

This tool relies on the common libraries listed in the main `requirements.txt` file at the project root.

-   `sounddevice`: For audio I/O.
-   `numpy`: For signal generation and analysis.
-   `rich`: For formatted output.

Ensure you have installed the common requirements:
```bash
# From the project root directory
pip install -r requirements.txt
```

## Usage

1.  **List available audio devices** to find the ID of the device you want to test.

    ```bash
    python3 audio_loopback_finder/audio_loopback_finder.py --list-devices
    ```

2.  **Run the finder** on your desired device ID using the `-d` or `--device` flag.

    ```bash
    python3 audio_loopback_finder/audio_loopback_finder.py -d <DEVICE_ID>
    ```

### Example Output

```
$ python3 audio_loopback_finder/audio_loopback_finder.py -d 0
Testing device: HDA Intel PCH: ALC269VC Analog (hw:0,0)
       Found Loopback Paths
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Output Channel ┃ Input Channel ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ 1              │ 2             │
└────────────────┴───────────────┘
```

## License

This tool is released into the public domain under the Unlicense.
