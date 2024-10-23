import sys
from src.logger import logging

def error_msg_details(err, error_details : sys):
    _, _, exc_tb = error_details.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    err_msg = "Error in the Python Script in file [{0}] at line [{1}] error meassge [{2}]".format(file_name, exc_tb.tb_lineno, str(err))

    return err_msg


class CustomException(Exception):
    def __init__(self, err, error_details : sys) -> None:
        super().__init__(err)
        self.err = error_msg_details(err, error_details)
    
    def __str__(self) -> str:
        return self.err
    

if __name__ == "__main__":
    try:
        a = 1/0
    except Exception as e:
        logging.info("Log saved")
        raise CustomException(e, sys)
        
