import sys 
import traceback  # Helpful for type hints and stack traces
import logging

def error_message_detail(error, error_detail):
    exc_type, exc_value, exc_tb = error_detail.exc_info()
    error_message = "Error occurred in python script name [unknown] line number [unknown] error message [{}]".format(str(error))
    
    if exc_tb is not None and exc_tb.tb_frame is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
            file_name, exc_tb.tb_lineno, str(error)
        )
 
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)  # ✅ Fixed: added ()
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
    
    def __str__(self):
        return self.error_message

