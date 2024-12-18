import logging
import os
import sys
import subprocess
from pathlib import Path

def setup_logging():
    # Get the script's directory and construct logs directory path
    script_dir = Path(__file__).resolve().parent
    logs_dir = script_dir.parent / 'logs'  # capture/logs directory
    
    # Create logs directory if it doesn't exist
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up log file path
    log_file = logs_dir / 'automation.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print("Starting network traffic capture and analysis pipeline...")
    
    try:
        # Get script directory and construct paths
        script_dir = Path(__file__).resolve().parent
        capture_script = script_dir / 'capture_traffic.py'
        process_script = script_dir / 'pcap_to_csv.py'
        
        # Validate scripts exist
        if not capture_script.exists():
            raise FileNotFoundError(f"Capture script not found: {capture_script}")
        if not process_script.exists():
            raise FileNotFoundError(f"Processing script not found: {process_script}")
        
        # Run capture_traffic.py
        print("\nStarting network traffic capture...")
        subprocess.run([sys.executable, str(capture_script)], check=True)
        
        # Run pcap_to_csv.py
        print("\nConverting PCAP to CSV...")
        subprocess.run([sys.executable, str(process_script)], check=True)
        
        print("\nPipeline completed successfully!")
            
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        print("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error("Pipeline failed: %s", str(e))
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
