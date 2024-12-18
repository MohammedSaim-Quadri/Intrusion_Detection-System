import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

def setup_logging():
    # Get the script's directory and construct logs directory path
    script_dir = Path(__file__).resolve().parent
    logs_dir = script_dir.parent / 'logs'  # capture/logs directory
    
    # Create logs directory if it doesn't exist
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up log file path
    log_file = logs_dir / 'cicflowmeter.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def get_unprocessed_pcaps(pcap_dir, output_dir):
    """Get list of PCAP files that haven't been processed yet."""
    # Get all pcap files
    pcap_files = set(Path(pcap_dir).glob('*.pcap'))
    if not pcap_files:
        logging.info("No PCAP files found in %s", pcap_dir)
        return []
    
    # Create a processed files tracking file
    processed_tracking_file = Path(output_dir) / '.processed_pcaps'
    processed_pcaps = set()
    
    # Read previously processed files
    if processed_tracking_file.exists():
        with open(processed_tracking_file, 'r') as f:
            processed_pcaps = set(line.strip() for line in f.readlines())
    
    # Get unprocessed files
    unprocessed = [pcap for pcap in pcap_files if str(pcap.absolute()) not in processed_pcaps]
    
    if unprocessed:
        logging.info("Found %d unprocessed PCAP files", len(unprocessed))
    else:
        logging.info("No new PCAP files to process")
    
    return sorted(unprocessed)

def mark_as_processed(pcap_file, output_dir):
    """Mark a PCAP file as processed."""
    tracking_file = Path(output_dir) / '.processed_pcaps'
    with open(tracking_file, 'a') as f:
        f.write(str(Path(pcap_file).absolute()) + '\n')

def run_cicflowmeter(pcap_file, output_path, cicflowmeter_path, max_retries=3):
    retry_count = 0
    while retry_count < max_retries:
        try:
            # Convert paths to absolute before changing directory
            pcap_file = str(Path(pcap_file).absolute())
            output_path = str(Path(output_path).absolute())
            cicflowmeter_path = Path(cicflowmeter_path).absolute()
            
            # Change to CICFlowMeter directory
            original_dir = os.getcwd()
            os.chdir(cicflowmeter_path)
            
            # Construct the Gradle command based on OS
            if os.name == 'nt':  # Windows
                gradle_cmd = str(cicflowmeter_path / 'gradlew.bat')
            else:  # Unix-like
                gradle_cmd = str(cicflowmeter_path / 'gradlew')
            
            # Construct the command
            cmd = [
                gradle_cmd,
                'exeCMD',
                '--args',
                f'"{pcap_file}" "{output_path}"'
            ]
            
            logging.info("Processing PCAP file: %s", pcap_file)
            logging.info("Running command: %s", ' '.join(cmd))
            
            # Run the command with a timeout
            try:
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    check=True,
                    timeout=300  # 5 minute timeout
                )
                
                if result.stdout:
                    logging.info("Output: %s", result.stdout)
                if result.stderr:
                    logging.warning("Stderr: %s", result.stderr)
                
                # Wait for file generation (up to 30 seconds)
                pcap_name = Path(pcap_file).stem
                expected_csv = None
                wait_time = 0
                while wait_time < 30:
                    # Check for any new CSV files
                    csv_files = list(Path(output_path).glob('*.csv'))
                    new_csvs = [f for f in csv_files if f.stat().st_mtime > time.time() - 60]
                    
                    if new_csvs:
                        expected_csv = new_csvs[0]  # Take the most recently created CSV
                        break
                        
                    time.sleep(1)
                    wait_time += 1
                
                if not expected_csv:
                    raise FileNotFoundError(f"No CSV file was generated in {output_path}")
                
                if expected_csv.stat().st_size == 0:
                    raise ValueError(f"Generated CSV file is empty: {expected_csv}")
                    
                logging.info("Successfully processed PCAP file. Output: %s", expected_csv)
                
                # Mark as processed only if successful
                mark_as_processed(pcap_file, output_path)
                return True
                
            except subprocess.TimeoutExpired:
                logging.error("Command timed out after 5 minutes")
                raise
                
        except subprocess.CalledProcessError as e:
            retry_count += 1
            logging.error("Attempt %d failed: %s", retry_count, str(e))
            if e.stdout: logging.error("Stdout: %s", e.stdout)
            if e.stderr: logging.error("Stderr: %s", e.stderr)
            if retry_count >= max_retries:
                raise RuntimeError(f"Failed to process PCAP file after {max_retries} attempts") from e
            
        except Exception as e:
            retry_count += 1
            logging.error("Attempt %d failed: %s", retry_count, str(e))
            if retry_count >= max_retries:
                raise
            
        finally:
            # Ensure we return to the original directory
            if 'original_dir' in locals():
                os.chdir(original_dir)
            
        # Wait before retrying
        if retry_count < max_retries:
            time.sleep(5)  # Wait 5 seconds before retrying
    
    return False

def main():
    setup_logging()

    try:
        # Get the IDS root directory (two levels up from the script)
        script_path = Path(__file__).resolve()
        root_dir = script_path.parents[2]  # From scripts/pcap_to_csv.py, go up to IDS root
        
        # Load settings from config directory
        config_path = root_dir / "config" / "settings.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Settings file not found: {config_path}")
            
        with open(config_path, "r", encoding='utf-8') as file:
            settings = yaml.safe_load(file)

        # Get absolute paths from settings, relative to root_dir
        pcap_dir = (root_dir / settings["pcap_path"]).resolve()
        output_path = (root_dir / settings["output_path"]).resolve()
        cicflowmeter_path = (root_dir / settings["cicflowmeter_path"]).resolve()
        max_retries = settings.get("max_retries", 3)

        # Validate paths
        if not pcap_dir.exists():
            raise FileNotFoundError(f"PCAP directory not found: {pcap_dir}")
        if not cicflowmeter_path.exists():
            raise FileNotFoundError(f"CICFlowMeter path not found: {cicflowmeter_path}")
        if not (cicflowmeter_path / ('gradlew.bat' if os.name == 'nt' else 'gradlew')).exists():
            raise FileNotFoundError(f"Gradle wrapper not found in {cicflowmeter_path}")
        
        # Create output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)

        # Get list of unprocessed PCAP files
        unprocessed_pcaps = get_unprocessed_pcaps(pcap_dir, output_path)
        
        if not unprocessed_pcaps:
            logging.info("No new PCAP files to process")
            print("No new PCAP files to process")
            return
            
        logging.info("Found %d unprocessed PCAP files", len(unprocessed_pcaps))
        print(f"Found {len(unprocessed_pcaps)} unprocessed PCAP files")
        
        # Process each PCAP file
        success_count = 0
        for pcap_file in unprocessed_pcaps:
            try:
                print(f"\nProcessing: {pcap_file.name}")
                if run_cicflowmeter(str(pcap_file), str(output_path), str(cicflowmeter_path), max_retries):
                    success_count += 1
                    print(f"Successfully processed: {pcap_file.name}")
                else:
                    print(f"Failed to process: {pcap_file.name}")
            except Exception as e:
                logging.error("Failed to process %s: %s", pcap_file.name, str(e))
                print(f"Error processing {pcap_file.name}: {e}")
                continue
        
        # Print summary
        print(f"\nProcessing complete: {success_count} of {len(unprocessed_pcaps)} files processed successfully")
        if success_count < len(unprocessed_pcaps):
            print("Some files failed to process. Check the log file for details.")

    except Exception as e:
        logging.error("Error during automation: %s", str(e))
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()