import logging
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

def setup_logging():
    # Get the script's directory and construct logs directory path
    script_dir = Path(__file__).resolve().parent
    logs_dir = script_dir.parent / 'logs'  # capture/logs directory
    
    # Create logs directory if it doesn't exist
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up log file path
    log_file = logs_dir / 'capture.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def get_interfaces():
    try:
        # First check if tshark is available
        try:
            subprocess.run(['tshark', '--version'], capture_output=True, check=True)
        except subprocess.CalledProcessError as exc:
            msg = "tshark is required but not found. Please install Wireshark/tshark first."
            logging.error(msg)
            raise RuntimeError(msg) from exc
            
        result = subprocess.run(['tshark', '-D'], capture_output=True, text=True, check=True)
        interfaces = result.stdout.strip().split('\n')
        
        # Filter out empty lines and strip whitespace
        interfaces = [iface.strip() for iface in interfaces if iface.strip()]
        
        if not interfaces:
            raise RuntimeError("No network interfaces found")
            
        return interfaces
    except subprocess.CalledProcessError as e:
        logging.error("Failed to get interfaces: %s", str(e))
        raise RuntimeError("Failed to list network interfaces") from e

def extract_interface_id(interface_string):
    # Extract just the NPF ID from the interface string
    match = re.search(r'\\Device\\NPF_\{([^}]+)\}', interface_string)
    if match:
        return f"\\Device\\NPF_{{{match.group(1)}}}"
    
    # If no NPF ID found, try to get the interface number
    match = re.match(r'(\d+)\.', interface_string)
    if match:
        return match.group(1)
        
    return interface_string

def validate_duration(duration):
    try:
        duration = int(duration)
        if duration <= 0:
            raise ValueError("Duration must be positive")
        if duration > 86400:  # 24 hours
            raise ValueError("Duration cannot exceed 24 hours")
        return duration
    except ValueError as e:
        logging.error("Invalid duration value: %s", str(e))
        raise

def start_packet_capture(interface, duration, output_dir):
    try:
        # Create output directory if it doesn't exist
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"capture_{timestamp}.pcap"
        
        # Extract the interface ID
        interface_id = extract_interface_id(interface)
        
        # Validate duration
        duration = validate_duration(duration)
        
        # Construct the command
        cmd = [
            'tshark',
            '-i', interface_id,
            '-a', f'duration:{duration}',
            '-w', str(output_file)
        ]
        
        logging.info("Starting capture on interface %s", interface)
        logging.info("Output file: %s", str(output_file))
        logging.info("Duration: %d seconds", duration)
        
        # Run tshark with output streaming
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        ) as process:
            # Monitor the process
            while True:
                return_code = process.poll()
                if return_code is not None:
                    break
                    
            if return_code != 0:
                stderr = process.stderr.read()
                raise subprocess.CalledProcessError(return_code, cmd, stderr=stderr)
                
            if not output_file.exists():
                raise RuntimeError(f"Capture completed but output file not found: {output_file}")
                
            file_size = output_file.stat().st_size
            if file_size == 0:
                raise RuntimeError(f"Capture completed but output file is empty: {output_file}")
                
            logging.info("Capture completed successfully: %s (%d bytes)", str(output_file), file_size)
            return str(output_file)
        
    except subprocess.CalledProcessError as e:
        logging.error("Capture failed: %s", str(e))
        if e.stderr:
            logging.error("Error output: %s", e.stderr)
        raise
    except Exception as e:
        logging.error("Unexpected error during capture: %s", str(e))
        raise

def main():
    setup_logging()
    
    try:
        # Get the IDS root directory (two levels up from the script)
        script_path = Path(__file__).resolve()
        root_dir = script_path.parents[2]  # From scripts/capture_traffic.py, go up to IDS root
        
        # Load settings from config directory
        config_path = root_dir / "config" / "settings.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Settings file not found: {config_path}")
            
        with open(config_path, "r", encoding='utf-8') as file:
            settings = yaml.safe_load(file)
            
        interface = settings.get('interface')
        duration = settings.get('duration', 600)
        
        # Get absolute path for output directory
        output_dir = (root_dir / settings['pcap_path']).resolve()
        
        # Get available interfaces
        interfaces = get_interfaces()
        
        # If interface not specified in settings, prompt user
        if not interface:
            print("\nAvailable interfaces:")
            for i, iface in enumerate(interfaces, 1):
                print(f"{i}. {iface}")
                
            while True:
                try:
                    choice = input("\nSelect interface number (or 'q' to quit): ").strip()
                    if choice.lower() == 'q':
                        sys.exit(0)
                        
                    choice = int(choice)
                    if 1 <= choice <= len(interfaces):
                        interface = interfaces[choice-1]
                        break
                    print(f"Please enter a number between 1 and {len(interfaces)}")
                except ValueError:
                    print("Please enter a valid number")
        
        # Start the capture
        output_file = start_packet_capture(interface, duration, output_dir)
        print("\nCapture completed successfully!")
        print(f"Output file: {output_file}")
        
    except KeyboardInterrupt:
        logging.info("Capture interrupted by user")
        print("\nCapture interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error("Error in main execution: %s", str(e))
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
